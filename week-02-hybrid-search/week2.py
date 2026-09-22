from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ollama
import streamlit as st
from rank_bm25 import BM25Okapi

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader


DEFAULT_LLM = "llama3.2"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
EMBED_BATCH_SIZE = 16
MAX_CONTEXT_CHARS = 12_000
RRF_K = 60
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@st.cache_data(ttl=20, show_spinner=False)
def ollama_status() -> Tuple[bool, List[str]]:
    try:
        resp = ollama.list()
        models = _get(resp, "models", []) or []
        names = []
        for m in models:
            name = _get(m, "model") or _get(m, "name")
            if name:
                names.append(name)
        return True, sorted(names)
    except Exception:
        return False, []


@st.cache_data(show_spinner=False, max_entries=10_000)
def get_embedding(text: str, model: str) -> List[float]:
    resp = ollama.embeddings(model=model, prompt=text)
    vec = _get(resp, "embedding", []) or []
    return [float(x) for x in vec]


def embed_texts(
    texts: List[str],
    model: str,
    progress_cb: Optional[Any] = None,
) -> List[List[float]]:
    vectors: List[List[float]] = []
    total = len(texts)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        vectors.extend(_embed_batch(batch, model))
        if progress_cb:
            progress_cb(min(start + EMBED_BATCH_SIZE, total), total)
    return vectors


def _embed_batch(batch: List[str], model: str) -> List[List[float]]:
    try:
        resp = ollama.embed(model=model, input=batch)
        embs = _get(resp, "embeddings", None)
        if embs is not None and len(embs) == len(batch):
            return [[float(x) for x in e] for e in embs]
    except Exception:
        pass
    return [get_embedding(t, model) for t in batch]


def stream_answer(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
):
    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options={"temperature": temperature},
    )
    for part in stream:
        piece = _get(_get(part, "message", {}), "content", "")
        if piece:
            yield piece


def extract_pages(uploaded_file) -> List[Dict[str, Any]]:
    reader = PdfReader(uploaded_file)
    pages: List[Dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    step = max(1, chunk_size - chunk_overlap)
    n = len(text)
    chunks: List[str] = []
    start = 0

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            window = text[start:end]
            for sep in ("\n\n", ". ", "! ", "? ", "\n", " "):
                idx = window.rfind(sep)
                if idx > chunk_size * 0.5:
                    end = start + idx + len(sep)
                    break

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= n:
            break
        start += step

    return chunks


def chunk_pages(
    pages: List[Dict[str, Any]],
    source_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0
    for page in pages:
        for piece in _chunk_text(page["text"], chunk_size, chunk_overlap):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": source_name,
                    "page": page["page"],
                    "text": piece,
                }
            )
            chunk_id += 1
    return chunks


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class HybridVectorStore:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None
        self._dirty: bool = True
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized: List[List[str]] = []

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embed_model: str,
        progress_cb: Optional[Any] = None,
    ) -> None:
        if not chunks:
            return

        vectors = embed_texts([c["text"] for c in chunks], embed_model, progress_cb)

        for chunk, vec in zip(chunks, vectors):
            self.records.append({"metadata": chunk, "vector": vec})
            self._tokenized.append(_tokenize(chunk["text"]))

        self._bm25 = BM25Okapi(self._tokenized)
        self._dirty = True

    def _build(self) -> None:
        if not self._dirty or not self.records:
            return
        mat = np.asarray([r["vector"] for r in self.records], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self._matrix = mat / norms
        self._dirty = False

    @property
    def size(self) -> int:
        return len(self.records)

    def _dense_scores(self, query_vec: List[float]) -> np.ndarray:
        self._build()
        q = np.asarray(query_vec, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-10
        return self._matrix @ q if self._matrix is not None else np.array([])

    def _dense_ranking(self, query_vec: List[float]) -> List[int]:
        sims = self._dense_scores(query_vec)
        if sims.size == 0:
            return []
        return list(np.argsort(-sims))

    def _sparse_ranking(self, query: str) -> List[int]:
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        return list(np.argsort(-scores))

    def retrieve_hybrid(
        self,
        query: str,
        query_vec: List[float],
        top_k: int = 4,
        rrf_k: int = RRF_K,
    ) -> List[Dict[str, Any]]:
        if not self.records:
            return []

        dense_ranking = self._dense_ranking(query_vec)
        sparse_ranking = self._sparse_ranking(query)

        fused: Dict[int, float] = {}
        for rank, idx in enumerate(dense_ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(sparse_ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        dense_sims = self._dense_scores(query_vec)

        results: List[Dict[str, Any]] = []
        for idx, fused_score in ordered[:top_k]:
            chunk = dict(self.records[idx]["metadata"])
            chunk["fused_score"] = float(fused_score)
            chunk["dense_score"] = float(dense_sims[idx]) if dense_sims.size else 0.0
            results.append(chunk)

        return results


def build_context(results: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    used = 0
    for r in results:
        block = (
            f"[Source {r['chunk_id']} | {r['source']} p.{r['page']}]\n{r['text']}"
        )
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_system_prompt(context: str) -> str:
    return (
        "You are a precise research assistant. Answer the user's question using ONLY "
        "the context provided below.\n"
        "Rules:\n"
        "1. If the answer is not contained in the context, reply exactly: "
        "'I cannot answer this based on the provided document.'\n"
        "2. Cite the sources you used inline, e.g. [Source 3].\n"
        "3. Be concise and do not invent details.\n\n"
        f"CONTEXT:\n{context}"
    )


st.set_page_config(
    page_title="Hybrid RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    :root {
        --bg: #0b0d12;
        --panel: #12151c;
        --panel-2: #171b24;
        --border: #232833;
        --text: #e6e8ee;
        --muted: #8a92a6;
        --accent: #5b8def;
        --accent-2: #7aa2ff;
        --good: #3ecf8e;
        --warn: #e0a94a;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter,
            Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 20% -10%, #161b26 0%, #0b0d12 55%);
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label {
        color: var(--muted);
        font-size: 0.85rem;
    }

    section[data-testid="stSidebar"] h3 {
        color: var(--text);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 1.2rem 0 0.6rem 0;
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 6rem;
        max-width: 1100px;
    }

    .app-title {
        font-size: 1.4rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--text);
        margin: 0;
    }

    .app-subtitle {
        color: var(--muted);
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(62, 207, 142, 0.08);
        border: 1px solid rgba(62, 207, 142, 0.28);
        color: var(--good);
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }

    .status-pill.offline {
        background: rgba(239, 68, 68, 0.08);
        border-color: rgba(239, 68, 68, 0.3);
        color: #f87171;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: currentColor;
        box-shadow: 0 0 6px currentColor;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.6rem;
    }

    .stat-card {
        background: linear-gradient(180deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }

    .stat-label {
        color: var(--muted);
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .stat-value {
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    div[data-testid="stChatMessage"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: var(--panel-2);
    }

    .stChatInput textarea {
        background: var(--panel-2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
    }

    .source-card {
        background: var(--panel-2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    }

    .source-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.45rem;
        gap: 0.6rem;
    }

    .source-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text);
    }

    .score-group {
        display: flex;
        gap: 0.35rem;
    }

    .score-badge {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.7rem;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        border: 1px solid transparent;
    }

    .score-fused {
        color: var(--accent-2);
        background: rgba(91, 141, 239, 0.08);
        border-color: rgba(91, 141, 239, 0.2);
    }

    .score-dense {
        color: var(--good);
        background: rgba(62, 207, 142, 0.08);
        border-color: rgba(62, 207, 142, 0.2);
    }

    .source-body {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        border: 1px dashed var(--border);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.01);
    }

    .empty-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.4rem;
    }

    .empty-body {
        color: var(--muted);
        font-size: 0.88rem;
    }

    .stButton > button {
        border-radius: 9px;
        border: 1px solid var(--border);
        background: var(--panel-2);
        color: var(--text);
        font-weight: 500;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent-2);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, #5b8def 0%, #4a7bdc 100%);
        border-color: #4a7bdc;
        color: #fff;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(180deg, #6a99f2 0%, #5488e6 100%);
        color: #fff;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        background: var(--panel);
    }

    div[data-testid="stExpander"] summary {
        color: var(--muted);
        font-size: 0.8rem;
    }

    .stProgress > div > div > div > div {
        background: var(--accent);
    }

    .stAlert {
        border-radius: 10px;
    }

    hr {
        border-color: var(--border);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = HybridVectorStore()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}

ollama_ok, available_models = ollama_status()

with st.sidebar:
    st.markdown("### Workspace")

    st.markdown("#### Models")
    llm_options = available_models or [DEFAULT_LLM]
    llm_model = st.selectbox(
        "Generation model",
        llm_options,
        index=llm_options.index(DEFAULT_LLM) if DEFAULT_LLM in llm_options else 0,
        label_visibility="collapsed",
    )
    embed_model = st.text_input(
        "Embedding model",
        value=DEFAULT_EMBED_MODEL,
        label_visibility="collapsed",
        placeholder="Embedding model",
    )

    st.markdown("#### Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("#### Chunking")
    chunk_size = st.slider("Chunk size", 200, 2000, 500, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 100, step=25)
    if chunk_overlap >= chunk_size:
        st.warning("Overlap clamped below chunk size.")
        chunk_overlap = max(0, chunk_size - 50)

    st.markdown("#### Retrieval")
    top_k = st.slider("Top-k chunks", 1, 10, 4)
    rrf_k = st.slider("RRF constant (k)", 10, 120, 60, step=5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, step=0.05)

    st.markdown("#### Actions")
    process_clicked = st.button(
        "Process Documents", type="primary", use_container_width=True
    )
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

status_class = "status-pill" if ollama_ok else "status-pill offline"
status_label = "Ollama Connected" if ollama_ok else "Ollama Offline"

header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown(
        """
        <div>
            <div class="app-title">Hybrid RAG</div>
            <div class="app-subtitle">Dense vector search fused with BM25 lexical ranking via RRF</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        f"""
        <div style="display:flex; justify-content:flex-end; padding-top:0.4rem;">
            <span class="{status_class}">
                <span class="status-dot"></span>{status_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

if not ollama_ok:
    st.error(
        "Cannot reach Ollama. Start the server with `ollama serve` and make sure "
        "the Python client can reach it."
    )
    st.stop()

if process_clicked:
    if not uploaded_files:
        st.warning("Upload at least one PDF before processing.")
    else:
        store = HybridVectorStore()
        all_chunks: List[Dict[str, Any]] = []
        page_count = 0

        with st.spinner("Extracting text..."):
            for uf in uploaded_files:
                try:
                    pages = extract_pages(uf)
                except Exception as exc:
                    st.error(f"Could not read {uf.name}: {exc}")
                    continue
                page_count += len(pages)
                all_chunks.extend(
                    chunk_pages(pages, uf.name, chunk_size, chunk_overlap)
                )

        if not all_chunks:
            st.error("No extractable text found. The PDFs may be scanned images.")
        else:
            progress = st.progress(0.0, text="Building dense and sparse indexes...")

            def _cb(done: int, total: int) -> None:
                progress.progress(
                    done / total, text=f"Embedding {done} of {total} chunks..."
                )

            try:
                store.add_chunks(all_chunks, embed_model, progress_cb=_cb)
            except Exception as exc:
                progress.empty()
                st.error(
                    f"Indexing failed: {exc}\n\n"
                    f"Try running `ollama pull {embed_model}`."
                )
                st.stop()

            progress.empty()

            st.session_state.vector_store = store
            st.session_state.document_processed = True
            st.session_state.messages = []
            st.session_state.doc_stats = {
                "files": len(uploaded_files),
                "pages": page_count,
                "chunks": len(all_chunks),
            }
            st.success(f"Indexed {len(all_chunks)} chunks from {page_count} pages.")


def render_sources(results: List[Dict[str, Any]]) -> None:
    for r in results:
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-head">
                    <div class="source-title">{r['source']} &middot; page {r['page']} &middot; chunk {r['chunk_id']}</div>
                    <div class="score-group">
                        <span class="score-badge score-fused">rrf {r.get('fused_score', 0):.4f}</span>
                        <span class="score-badge score-dense">cos {r.get('dense_score', 0):.3f}</span>
                    </div>
                </div>
                <div class="source-body">{r['text'][:700]}{'…' if len(r['text']) > 700 else ''}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if st.session_state.document_processed:
    stats = st.session_state.doc_stats
    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Documents</div>
                <div class="stat-value">{stats.get('files', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Pages</div>
                <div class="stat-value">{stats.get('pages', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Chunks</div>
                <div class="stat-value">{stats.get('chunks', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Embedding Model</div>
                <div class="stat-value">{embed_model.split(':')[0]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Retrieved Context"):
                    render_sources(msg["sources"])

    prompt = st.chat_input("Ask a question about your documents...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Running hybrid retrieval..."):
                try:
                    query_vec = get_embedding(prompt, embed_model)
                    results = st.session_state.vector_store.retrieve_hybrid(
                        prompt, query_vec, top_k=top_k, rrf_k=rrf_k
                    )
                except Exception as exc:
                    st.error(f"Retrieval failed: {exc}")
                    st.stop()

            context = build_context(results)
            convo = [
                {"role": "system", "content": build_system_prompt(context)},
                *[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-6:]
                ],
            ]

            placeholder = st.empty()
            answer = ""
            try:
                for token in stream_answer(convo, llm_model, temperature):
                    answer += token
                    placeholder.markdown(answer + "▌")
                placeholder.markdown(answer or "_No response._")
            except Exception as exc:
                placeholder.empty()
                answer = f"Generation failed: {exc}"
                st.error(answer)

            if results:
                with st.expander("Retrieved Context"):
                    render_sources(results)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": results}
        )

elif not uploaded_files:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-title">No documents loaded</div>
            <div class="empty-body">
                Upload one or more PDFs from the sidebar, then click Process Documents to begin.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )