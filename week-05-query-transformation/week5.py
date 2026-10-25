import argparse
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ollama
import streamlit as st

try:
    from pypdf import PdfReader
except ImportError:
    import PyPDF2
    PdfReader = PyPDF2.PdfReader


DEFAULT_EMBED_MODEL = "nomic-embed-text"
PREFERRED_LLM = ["llama3.2", "llama3.1", "llama3", "mistral", "qwen2.5", "phi3"]
TOP_K = 4


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@st.cache_data(ttl=20, show_spinner=False)
def list_ollama_models() -> List[str]:
    try:
        resp = ollama.list()
        models = _get(resp, "models", []) or []
        names = []
        for m in models:
            name = _get(m, "model") or _get(m, "name")
            if name:
                names.append(name)
        return sorted(names)
    except Exception:
        return []


def pick_model(available: List[str], preferred: List[str]) -> Optional[str]:
    for candidate in preferred:
        for name in available:
            if name == candidate or name.split(":")[0] == candidate:
                return name
    return available[0] if available else None


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def chunk_text(
    text: str,
    source_name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return []

    step = max(1, chunk_size - chunk_overlap)
    n = len(text)
    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_id = 0

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
            chunks.append({
                "chunk_id": f"{source_name}::{chunk_id}",
                "source": source_name,
                "text": piece,
            })
            chunk_id += 1

        if end >= n:
            break
        start += step

    return chunks


@st.cache_data(show_spinner=False, max_entries=10_000)
def get_embedding(text: str, model: str) -> List[float]:
    response = ollama.embeddings(model=model, prompt=text)
    vec = _get(response, "embedding", []) or []
    return [float(x) for x in vec]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a, b = np.array(v1), np.array(v2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


class VectorStore:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def clear(self) -> None:
        self.records = []

    def add_chunks(self, chunks: List[Dict[str, Any]], embed_model: str, progress_cb=None) -> None:
        total = len(chunks)
        for i, c in enumerate(chunks):
            c["vector"] = get_embedding(c["text"], embed_model)
            self.records.append(c)
            if progress_cb:
                progress_cb(i + 1, total)

    def search(self, query_vec: List[float], top_k: int = TOP_K) -> List[Tuple[Dict[str, Any], float]]:
        scored = []
        for item in self.records:
            sim = cosine_similarity(query_vec, item["vector"])
            scored.append((item, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def generate_multi_queries(original_query: str, model: str, count: int = 3) -> List[str]:
    prompt = f"""You are an AI language model assistant. Generate {count} different versions of the user's question to retrieve relevant documents from a vector database. Each version should explore a different angle, use different vocabulary, or invert the phrasing. The goal is to overcome the limitations of distance-based similarity search.

Return each question on its own line. Do not number them. Do not add introductory text.

Original question: {original_query}"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.4},
    )
    raw = _get(_get(response, "message", {}), "content", "")
    lines = [line.strip(" -\t") for line in raw.split("\n") if line.strip()]
    return [original_query] + lines[:count]


def generate_hyde_document(original_query: str, model: str) -> str:
    prompt = f"""Write a short paragraph answering the following question. Do not worry about whether the information is factually accurate. Write a plausible-sounding answer that uses the industry vocabulary, tone, and terminology one would expect to find in a real document answering this question. Do not include any introductory filler. Write only the paragraph.

Question: {original_query}"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.6},
    )
    return _get(_get(response, "message", {}), "content", "").strip()


def stream_answer(query: str, context: str, model: str, temperature: float = 0.2):
    system = (
        "You are an expert assistant. Answer the user's question using ONLY the context below. "
        "Always cite your sources using the chunk id (e.g. [Source file.pdf::3]). "
        "If the answer is not in the context, say 'I cannot answer this based on the provided document.'"
    )
    user = f"Context:\n{context}\n\nQuestion: {query}"

    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True,
        options={"temperature": temperature},
    )
    for part in stream:
        piece = _get(_get(part, "message", {}), "content", "")
        if piece:
            yield piece


def dedupe_and_sort(
    chunks: List[Tuple[Dict[str, Any], float]], top_k: int = TOP_K
) -> List[Tuple[Dict[str, Any], float]]:
    best: Dict[str, Tuple[Dict[str, Any], float]] = {}
    for chunk, score in chunks:
        cid = chunk["chunk_id"]
        if cid not in best or score > best[cid][1]:
            best[cid] = (chunk, score)
    ordered = sorted(best.values(), key=lambda x: x[1], reverse=True)
    return ordered[:top_k]


def retrieve_standard(store: VectorStore, query: str, embed_model: str, top_k: int = TOP_K):
    q_vec = get_embedding(query, embed_model)
    results = store.search(q_vec, top_k=top_k)
    return results, {"queries": [query], "hyde": None}


def retrieve_multi_query(
    store: VectorStore, query: str, embed_model: str, llm_model: str,
    n_queries: int = 3, per_query_k: int = 2, top_k: int = TOP_K,
):
    queries = generate_multi_queries(query, llm_model, count=n_queries)
    pooled: List[Tuple[Dict[str, Any], float]] = []
    for q in queries:
        q_vec = get_embedding(q, embed_model)
        pooled.extend(store.search(q_vec, top_k=per_query_k))
    results = dedupe_and_sort(pooled, top_k=top_k)
    return results, {"queries": queries, "hyde": None}


def retrieve_hyde(
    store: VectorStore, query: str, embed_model: str, llm_model: str, top_k: int = TOP_K
):
    hypothetical = generate_hyde_document(query, llm_model)
    hyde_vec = get_embedding(hypothetical, embed_model)
    results = store.search(hyde_vec, top_k=top_k)
    return results, {"queries": [query], "hyde": hypothetical}


def build_context(results: List[Tuple[Dict[str, Any], float]]) -> str:
    blocks = []
    for chunk, _score in results:
        blocks.append(f"[Source {chunk['chunk_id']}]:\n{chunk['text']}")
    return "\n\n".join(blocks)


st.set_page_config(
    page_title="Query Transformation",
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
        --bad: #f87171;
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
        padding-bottom: 5rem;
        max-width: 1200px;
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
        color: var(--bad);
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: currentColor;
        box-shadow: 0 0 6px currentColor;
    }

    .strategy-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.6rem 0 1.4rem 0;
    }

    .strategy-card {
        background: linear-gradient(180deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }

    .strategy-card.active {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px rgba(91, 141, 239, 0.35);
    }

    .strategy-step {
        color: var(--accent-2);
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .strategy-name {
        color: var(--text);
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .strategy-desc {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.5;
    }

    .strategy-value {
        color: var(--accent-2);
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .answer-card {
        background: linear-gradient(180deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1.2rem;
    }

    .answer-label {
        color: var(--accent-2);
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .answer-body {
        color: var(--text);
        font-size: 0.94rem;
        line-height: 1.65;
        white-space: pre-wrap;
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
        margin-bottom: 0.5rem;
        gap: 0.6rem;
        flex-wrap: wrap;
    }

    .source-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text);
    }

    .source-score {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.7rem;
        color: var(--accent-2);
        background: rgba(91, 141, 239, 0.08);
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        border: 1px solid rgba(91, 141, 239, 0.25);
    }

    .source-body {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .query-chip {
        display: inline-block;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.75rem;
        color: var(--text);
        background: var(--panel-2);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.35rem 0.6rem;
        margin: 0 0.35rem 0.35rem 0;
    }

    .query-chip.original {
        color: var(--accent-2);
        border-color: rgba(91, 141, 239, 0.35);
        background: rgba(91, 141, 239, 0.08);
    }

    .hyde-block {
        background: var(--panel-2);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        color: var(--muted);
        font-size: 0.85rem;
        line-height: 1.6;
        white-space: pre-wrap;
        margin-bottom: 1rem;
    }

    .empty-state {
        text-align: center;
        padding: 3.5rem 2rem;
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
        font-size: 0.82rem;
    }

    .stProgress > div > div > div > div {
        background: var(--accent);
    }

    .stAlert {
        border-radius: 10px;
    }

    .stTextInput input {
        background: var(--panel-2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }

    hr {
        border-color: var(--border);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

available_models = list_ollama_models()
ollama_ok = len(available_models) > 0
default_llm = pick_model(available_models, PREFERRED_LLM) or "llama3.2"

STRATEGIES = [
    ("standard", "Standard", "Single query, embed and search as-is."),
    ("multi_query", "Multi-Query", "Rewrite into N angles, pool and dedupe the results."),
    ("hyde", "HyDE", "Hallucinate a hypothetical answer, embed that instead."),
]

with st.sidebar:
    st.markdown("### Workspace")

    st.markdown("#### Models")
    if ollama_ok:
        llm_model = st.selectbox(
            "LLM model",
            available_models,
            index=available_models.index(default_llm),
            label_visibility="collapsed",
        )
    else:
        llm_model = default_llm
        st.caption("No models detected")

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

    st.markdown("#### Actions")
    process_clicked = st.button(
        "Process Documents", type="primary", use_container_width=True
    )
    compare_mode = st.toggle(
        "Compare all strategies", value=False,
        help="Run Standard, Multi-Query, and HyDE side by side for the same query."
    )
    if st.button("Clear Index", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.indexed_docs = []
        st.session_state.last_result = None
        st.rerun()

    st.markdown("#### Index")
    st.caption(
        f"{len(st.session_state.indexed_docs)} documents · "
        f"{len(st.session_state.vector_store.records)} chunks"
    )


status_class = "status-pill" if ollama_ok else "status-pill offline"
status_label = "Ollama Connected" if ollama_ok else "Ollama Offline"

header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown(
        """
        <div>
            <div class="app-title">Query Transformation</div>
            <div class="app-subtitle">Multi-Query expansion and HyDE over a local vector index</div>
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
        "Cannot reach Ollama, or no models are installed. "
        "Start the server with `ollama serve`, then pull a model such as "
        "`ollama pull llama3.2` and refresh this page."
    )
    st.stop()

if process_clicked:
    if not uploaded_files:
        st.warning("Upload at least one PDF before processing.")
    else:
        all_chunks: List[Dict[str, Any]] = []
        indexed_names: List[str] = []
        errors: List[str] = []

        extraction_bar = st.progress(0.0, text="Extracting text...")
        for i, uf in enumerate(uploaded_files):
            try:
                raw_text = extract_text_from_pdf(uf)
                chunks = chunk_text(raw_text, uf.name, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
                indexed_names.append(uf.name)
            except Exception as exc:
                errors.append(f"{uf.name}: {exc}")
            extraction_bar.progress(
                (i + 1) / len(uploaded_files),
                text=f"Extracting {i + 1} of {len(uploaded_files)}...",
            )
        extraction_bar.empty()

        for err in errors:
            st.warning(f"Could not read {err}")

        if not all_chunks:
            st.error("No extractable text found in the uploaded PDFs.")
        else:
            embed_bar = st.progress(0.0, text="Generating embeddings...")

            def _cb(done: int, total: int) -> None:
                embed_bar.progress(done / total, text=f"Embedding {done} of {total}...")

            st.session_state.vector_store.clear()
            try:
                st.session_state.vector_store.add_chunks(all_chunks, embed_model, _cb)
            except Exception as exc:
                embed_bar.empty()
                st.error(
                    f"Embedding failed: {exc}. "
                    f"Try running `ollama pull {embed_model}`."
                )
                st.stop()
            embed_bar.empty()

            st.session_state.indexed_docs = indexed_names
            st.session_state.last_result = None
            st.success(f"Indexed {len(all_chunks)} chunks from {len(indexed_names)} documents.")


has_index = len(st.session_state.vector_store.records) > 0

if has_index:
    st.markdown("#### Strategy")
    strategy_cols = st.columns(3)
    selected_strategy = st.session_state.get("strategy_choice", "standard")

    for col, (key, name, desc) in zip(strategy_cols, STRATEGIES):
        with col:
            active = "active" if key == selected_strategy and not compare_mode else ""
            st.markdown(
                f"""
                <div class="strategy-card {active}">
                    <div class="strategy-step">Strategy</div>
                    <div class="strategy-name">{name}</div>
                    <div class="strategy-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not compare_mode:
        selected_strategy = st.radio(
            "Select strategy",
            [k for k, _, _ in STRATEGIES],
            format_func=lambda k: next(n for kk, n, _ in STRATEGIES if kk == k),
            horizontal=True,
            label_visibility="collapsed",
            key="strategy_choice",
        )

    st.markdown("#### Query")
    query = st.text_input(
        "Ask a vague or conversational question",
        placeholder="e.g. How does the company help with remote work?",
        label_visibility="collapsed",
    )

    if st.button("Run", type="primary") and query:
        results_by_strategy: Dict[str, Dict[str, Any]] = {}
        strategies_to_run = (
            [k for k, _, _ in STRATEGIES] if compare_mode else [selected_strategy]
        )

        for key in strategies_to_run:
            with st.spinner(f"Running {key}..."):
                try:
                    if key == "standard":
                        results, meta = retrieve_standard(
                            st.session_state.vector_store, query, embed_model
                        )
                    elif key == "multi_query":
                        results, meta = retrieve_multi_query(
                            st.session_state.vector_store, query, embed_model, llm_model
                        )
                    else:
                        results, meta = retrieve_hyde(
                            st.session_state.vector_store, query, embed_model, llm_model
                        )
                    results_by_strategy[key] = {"results": results, "meta": meta}
                except Exception as exc:
                    st.error(f"{key} failed: {exc}")

        st.session_state.last_result = {
            "query": query,
            "by_strategy": results_by_strategy,
            "compare": compare_mode,
        }

    if st.session_state.last_result:
        payload = st.session_state.last_result
        by_strategy = payload["by_strategy"]

        if payload["compare"]:
            cols = st.columns(len(by_strategy))
            for col, (key, data) in zip(cols, by_strategy.items()):
                name = next(n for kk, n, _ in STRATEGIES if kk == key)
                with col:
                    st.markdown(f"##### {name}")
                    meta = data["meta"]
                    if meta.get("hyde"):
                        st.markdown("**HyDE document**")
                        st.markdown(
                            f"<div class='hyde-block'>{meta['hyde']}</div>",
                            unsafe_allow_html=True,
                        )
                    if key == "multi_query":
                        chips = "".join(
                            f"<span class='query-chip{' original' if i == 0 else ''}'>{q}</span>"
                            for i, q in enumerate(meta["queries"])
                        )
                        st.markdown("**Generated queries**")
                        st.markdown(chips, unsafe_allow_html=True)
                    st.markdown("**Top chunks**")
                    for chunk, score in data["results"]:
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-head">
                                    <div class="source-title">{chunk['chunk_id']}</div>
                                    <div class="source-score">{score:.3f}</div>
                                </div>
                                <div class="source-body">{chunk['text'][:220]}{'…' if len(chunk['text']) > 220 else ''}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        else:
            key = next(iter(by_strategy.keys()))
            data = by_strategy[key]
            meta = data["meta"]

            if meta.get("hyde"):
                st.markdown("**HyDE document**")
                st.markdown(
                    f"<div class='hyde-block'>{meta['hyde']}</div>",
                    unsafe_allow_html=True,
                )

            if key == "multi_query":
                chips = "".join(
                    f"<span class='query-chip{' original' if i == 0 else ''}'>{q}</span>"
                    for i, q in enumerate(meta["queries"])
                )
                st.markdown("**Generated queries**")
                st.markdown(chips, unsafe_allow_html=True)

            context_str = build_context(data["results"])

            placeholder = st.empty()
            answer = ""
            try:
                for token in stream_answer(query=payload["query"], context=context_str, model=llm_model):
                    answer += token
                    placeholder.markdown(answer + "▌")
                placeholder.markdown(answer or "_No response._")
            except Exception as exc:
                placeholder.empty()
                st.error(f"Generation failed: {exc}")

            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="answer-label">Answer</div>
                    <div class="answer-body">{answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            ) if answer else None

            with st.expander("Retrieved Context", expanded=False):
                for chunk, score in data["results"]:
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-head">
                                <div class="source-title">{chunk['chunk_id']}</div>
                                <div class="source-score">{score:.3f}</div>
                            </div>
                            <div class="source-body">{chunk['text'][:700]}{'…' if len(chunk['text']) > 700 else ''}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-title">No documents indexed</div>
            <div class="empty-body">
                Upload one or more PDFs from the sidebar, then click Process Documents to begin.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )