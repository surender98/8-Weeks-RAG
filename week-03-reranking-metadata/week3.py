import re
import pandas as pd
import streamlit as st
import numpy as np
import PyPDF2
import ollama
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder


@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


reranker_model = load_reranker()


@st.cache_data(ttl=20, show_spinner=False)
def list_ollama_models() -> List[str]:
    try:
        resp = ollama.list()
        models = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", [])
        names = []
        for m in models or []:
            name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            if not name:
                name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
            if name:
                names.append(name)
        return sorted(names)
    except Exception:
        return []


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def chunk_text(text: str, doc_name: str, category: str, year: int, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    chunks = []
    start = 0
    chunk_id = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length and text[end] not in [" ", "\n", ".", ","]:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append({
                "chunk_id": f"{doc_name}_{chunk_id}",
                "text": chunk_content,
                "metadata": {
                    "category": category,
                    "year": year,
                    "source": doc_name
                }
            })
            chunk_id += 1

        start += (chunk_size - chunk_overlap)
        if end == text_length:
            break
    return chunks


def get_embedding(text: str) -> List[float]:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a, b = np.array(v1), np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TwoStageVectorStore:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Dict[str, Any]], progress_cb=None):
        total = len(chunks)
        for i, c in enumerate(chunks):
            c["vector"] = get_embedding(c["text"])
            self.records.append(c)
            if progress_cb:
                progress_cb(i + 1, total)

    def clear(self):
        self.records = []

    def retrieve_and_rerank(self, query: str, filter_category: str = "All", filter_year_min: int = 2000, top_n_initial: int = 20, top_k_final: int = 3):
        filtered_records = []
        for r in self.records:
            match_cat = (filter_category == "All" or r["metadata"]["category"] == filter_category)
            match_year = (r["metadata"]["year"] >= filter_year_min)
            if match_cat and match_year:
                filtered_records.append(r)

        if not filtered_records:
            return [], 0, []

        query_vec = get_embedding(query)
        scored_chunks = []
        for item in filtered_records:
            sim = cosine_similarity(query_vec, item["vector"])
            scored_chunks.append((item, sim))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        candidate_chunks = scored_chunks[:top_n_initial]

        cross_inp = [[query, chunk["text"]] for chunk, _ in candidate_chunks]
        cross_scores = reranker_model.predict(cross_inp)

        reranked_results = []
        for i in range(len(candidate_chunks)):
            reranked_results.append({
                "chunk": candidate_chunks[i][0],
                "bi_encoder_score": candidate_chunks[i][1],
                "cross_encoder_score": cross_scores[i]
            })

        reranked_results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        return reranked_results[:top_k_final], len(filtered_records), candidate_chunks


CATEGORIES = ["HR Policy", "Engineering", "Financials", "Legal"]

CATEGORY_KEYWORDS = [
    (["hr", "handbook", "people", "customer_success", "onboarding"], "HR Policy"),
    (["engineering", "product", "roadmap", "platform", "architecture"], "Engineering"),
    (["financial", "finance", "sales", "playbook", "revenue", "pricing"], "Financials"),
    (["legal", "security", "compliance", "privacy", "policy"], "Legal"),
]


def infer_category(filename: str) -> str:
    lower = filename.lower()
    for keywords, category in CATEGORY_KEYWORDS:
        if any(k in lower for k in keywords):
            return category
    return "HR Policy"


def infer_year(filename: str) -> int:
    match = re.search(r"(20\d{2})", filename)
    if match:
        return int(match.group(1))
    return 2024


st.set_page_config(
    page_title="Two-Stage RAG",
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
        max-width: 1150px;
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

    .pipeline {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 1.2rem 0 1.6rem 0;
    }

    .pipe-card {
        background: linear-gradient(180deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }

    .pipe-step {
        color: var(--accent-2);
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .pipe-name {
        color: var(--text);
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .pipe-desc {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.5;
    }

    .pipe-value {
        color: var(--accent-2);
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.85rem;
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
        padding: 0.9rem 1.05rem;
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

    .source-meta {
        color: var(--muted);
        font-size: 0.72rem;
        margin-top: 0.15rem;
    }

    .score-group {
        display: flex;
        gap: 0.35rem;
        flex-wrap: wrap;
    }

    .score-badge {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.7rem;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        border: 1px solid transparent;
    }

    .score-cross {
        color: var(--accent-2);
        background: rgba(91, 141, 239, 0.08);
        border-color: rgba(91, 141, 239, 0.25);
    }

    .score-bi {
        color: var(--good);
        background: rgba(62, 207, 142, 0.08);
        border-color: rgba(62, 207, 142, 0.2);
    }

    .source-body {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
        margin-top: 0.5rem;
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

    .stTextInput input,
    .stNumberInput input {
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
    st.session_state.vector_store = TwoStageVectorStore()
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

available_models = list_ollama_models()
ollama_ok = len(available_models) > 0

PREFERRED_MODEL = "llama3.2"
default_index = available_models.index(PREFERRED_MODEL) if PREFERRED_MODEL in available_models else 0

with st.sidebar:
    st.markdown("### Workspace")

    st.markdown("#### Model")
    if ollama_ok:
        selected_model = st.selectbox(
            "Generation model",
            available_models,
            index=default_index,
            label_visibility="collapsed",
        )
    else:
        selected_model = PREFERRED_MODEL
        st.caption("No models detected")

    st.markdown("#### Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("#### Index")
    st.caption(f"{len(st.session_state.indexed_docs)} documents · {len(st.session_state.vector_store.records)} chunks")

    if st.button("Clear Index", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.indexed_docs = []
        st.rerun()


status_class = "status-pill" if ollama_ok else "status-pill offline"
status_label = "Ollama Connected" if ollama_ok else "Ollama Offline"

header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown(
        """
        <div>
            <div class="app-title">Two-Stage RAG</div>
            <div class="app-subtitle">Metadata pre-filter, wide bi-encoder recall, precise cross-encoder reranking</div>
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

if uploaded_files:
    st.markdown("#### Tag Uploaded Documents")
    st.caption("Category and year are inferred from filenames. Edit any cell to override before processing.")

    rows = []
    for f in uploaded_files:
        rows.append({
            "file": f.name,
            "category": infer_category(f.name),
            "year": infer_year(f.name),
        })
    df = pd.DataFrame(rows)

    edited = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        disabled=["file"],
        column_config={
            "file": st.column_config.TextColumn("Document", width="large"),
            "category": st.column_config.SelectboxColumn(
                "Category", options=CATEGORIES, required=True, width="medium"
            ),
            "year": st.column_config.NumberColumn(
                "Year", min_value=2000, max_value=2030, step=1, required=True, width="small"
            ),
        },
        key="tag_editor",
    )

    st.markdown("#### Chunking Settings")
    chunk_col1, chunk_col2 = st.columns(2)
    with chunk_col1:
        chunk_size = st.slider("Chunk size (chars)", 200, 2000, 500, step=50)
    with chunk_col2:
        chunk_overlap = st.slider("Chunk overlap (chars)", 0, 500, 100, step=25)
    if chunk_overlap >= chunk_size:
        st.warning("Overlap clamped below chunk size.")
        chunk_overlap = max(0, chunk_size - 50)

    process_clicked = st.button(
        "Process All Documents", type="primary", use_container_width=False
    )

    if process_clicked:
        all_chunks: List[Dict[str, Any]] = []
        indexed_names: List[str] = []
        errors: List[str] = []

        extraction_bar = st.progress(0.0, text="Extracting text...")
        for i, (f, row) in enumerate(zip(uploaded_files, edited.itertuples(index=False))):
            try:
                raw_text = extract_text_from_pdf(f)
                chunks = chunk_text(
                    raw_text,
                    f.name,
                    row.category,
                    int(row.year),
                    chunk_size,
                    chunk_overlap,
                )
                all_chunks.extend(chunks)
                indexed_names.append(f.name)
            except Exception as exc:
                errors.append(f"{f.name}: {exc}")
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
                embed_bar.progress(done / total, text=f"Embedding {done} of {total} chunks...")

            st.session_state.vector_store.clear()
            st.session_state.vector_store.add_chunks(all_chunks, progress_cb=_cb)
            embed_bar.empty()

            st.session_state.indexed_docs = indexed_names
            st.success(
                f"Indexed {len(all_chunks)} chunks from {len(indexed_names)} documents."
            )

has_index = len(st.session_state.vector_store.records) > 0

if has_index:
    st.markdown("#### Search Filters")
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
    with filter_col1:
        search_category = st.selectbox("Category", ["All"] + CATEGORIES)
    with filter_col2:
        search_year = st.slider("Min year", 2000, 2025, 2000)
    with filter_col3:
        top_k_final = st.slider("Final chunks (top-k)", 1, 10, 3)

    query = st.text_input("Ask a question", placeholder="e.g. What is the parental leave policy?")

    generate_clicked = st.button("Generate Answer", type="primary")

    if generate_clicked and query:
        with st.spinner("Filtering, retrieving, reranking, generating..."):
            top_results, filter_count, candidate_set = st.session_state.vector_store.retrieve_and_rerank(
                query=query,
                filter_category=search_category,
                filter_year_min=search_year,
                top_n_initial=15,
                top_k_final=top_k_final,
            )

        if not top_results:
            st.error("No documents matched your metadata filters.")
        else:
            context_blocks = []
            for res in top_results:
                chunk = res["chunk"]
                context_blocks.append(
                    f"[Source {chunk['chunk_id']} | {chunk['metadata']['category']} {chunk['metadata']['year']}]:\n{chunk['text']}"
                )
            context_str = "\n\n".join(context_blocks)

            system_prompt = (
                "You are an expert assistant. Answer the user's question using ONLY the context provided below.\n"
                "Always cite your sources using the Source ID provided.\n\n"
                f"CONTEXT:\n{context_str}"
            )

            try:
                response = ollama.chat(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                )
                answer_text = response["message"]["content"]
            except ollama.ResponseError as exc:
                st.error(
                    f"Model `{selected_model}` failed to respond: {exc}. "
                    f"Pull it with `ollama pull {selected_model}` or pick a different model in the sidebar."
                )
                st.stop()
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                st.stop()

            st.markdown(
                f"""
                <div class="pipeline">
                    <div class="pipe-card">
                        <div class="pipe-step">Stage 1</div>
                        <div class="pipe-name">Metadata Pre-Filter</div>
                        <div class="pipe-desc">Narrow the corpus before any vector math runs.</div>
                        <div class="pipe-value">{filter_count} / {len(st.session_state.vector_store.records)} chunks</div>
                    </div>
                    <div class="pipe-card">
                        <div class="pipe-step">Stage 2</div>
                        <div class="pipe-name">Bi-Encoder Recall</div>
                        <div class="pipe-desc">Fast dense vector search cast a wide net over the filtered set.</div>
                        <div class="pipe-value">{len(candidate_set)} candidates</div>
                    </div>
                    <div class="pipe-card">
                        <div class="pipe-step">Stage 3</div>
                        <div class="pipe-name">Cross-Encoder Rerank</div>
                        <div class="pipe-desc">Precise joint query-document scoring across every candidate.</div>
                        <div class="pipe-value">{len(top_results)} final chunks</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="answer-label">Answer</div>
                    <div class="answer-body">{answer_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Retrieved Context", expanded=False):
                for res in top_results:
                    chunk = res["chunk"]
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-head">
                                <div>
                                    <div class="source-title">{chunk['metadata']['source']} &middot; chunk {chunk['chunk_id']}</div>
                                    <div class="source-meta">{chunk['metadata']['category']} &middot; {chunk['metadata']['year']}</div>
                                </div>
                                <div class="score-group">
                                    <span class="score-badge score-cross">cross {res['cross_encoder_score']:.4f}</span>
                                    <span class="score-badge score-bi">bi {res['bi_encoder_score']:.3f}</span>
                                </div>
                            </div>
                            <div class="source-body">{chunk['text'][:700]}{'…' if len(chunk['text']) > 700 else ''}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander("Reranking Detail", expanded=False):
                st.markdown(
                    "Bi-encoder scores the query and each document independently for speed. "
                    "The cross-encoder reads the query and document together, which is slower "
                    "but far more accurate. That's why it only runs on the pre-filtered top candidates."
                )
                st.divider()
                for res in top_results:
                    chunk = res["chunk"]
                    st.markdown(
                        f"**{chunk['chunk_id']}** &nbsp;&middot;&nbsp; "
                        f"bi-encoder `{res['bi_encoder_score']:.4f}` &nbsp;&rarr;&nbsp; "
                        f"cross-encoder `{res['cross_encoder_score']:.4f}`"
                    )
                    st.text(chunk["text"])
                    st.divider()

else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-title">No documents indexed</div>
            <div class="empty-body">
                Upload your PDFs from the sidebar. Tag them with a category and year, then click Process All Documents.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )