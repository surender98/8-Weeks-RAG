import html
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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
MAX_AGENT_STEPS = 5
TOP_K = 3
MAX_OBSERVATION_CHARS = 3000
MAX_TRACE_PREVIEW_CHARS = 600
MAX_WEB_QUERY_CHARS = 400
MAX_WEB_RESULTS = 3
WEB_RETRY_DELAY = 0.8


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
    resp = ollama.embeddings(model=model, prompt=text)
    vec = _get(resp, "embedding", []) or []
    return [float(x) for x in vec]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a, b = np.array(v1), np.array(v2)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


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

    def search(self, query: str, embed_model: str, top_k: int = TOP_K) -> str:
        if not self.records:
            return "Internal database is empty. No documents are loaded."
        if not query.strip():
            return "Internal search failed: empty query."
        q_vec = get_embedding(query, embed_model)
        scored = [(item, cosine_similarity(q_vec, item["vector"])) for item in self.records]
        scored.sort(key=lambda x: x[1], reverse=True)
        lines = []
        for chunk, score in scored[:top_k]:
            lines.append(f"[{chunk['chunk_id']} | score {score:.3f}]\n{chunk['text']}")
        return "\n\n".join(lines)


@dataclass
class TraceStep:
    kind: str
    content: str
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    elapsed_s: float = 0.0


def tool_search_internal(query: str, vector_store: VectorStore, embed_model: str) -> str:
    return vector_store.search(query, embed_model)


def _ddg_search(query: str, max_results: int = MAX_WEB_RESULTS) -> List[Dict[str, Any]]:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError(
                "Install `ddgs` (recommended) or `duckduckgo-search` "
                "to enable live web search."
            )

    with DDGS() as client:
        return list(client.text(query, max_results=max_results))


def tool_search_web(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return "Web search failed: empty query."

    if len(query) > MAX_WEB_QUERY_CHARS:
        query = query[:MAX_WEB_QUERY_CHARS].rsplit(" ", 1)[0]

    results: List[Dict[str, Any]] = []
    last_error: Optional[str] = None

    for attempt in range(2):
        try:
            results = _ddg_search(query)
            if results:
                break
        except ImportError as exc:
            return f"Web search unavailable: {exc}"
        except Exception as exc:
            last_error = str(exc)

        if attempt == 0:
            time.sleep(WEB_RETRY_DELAY)

    if not results:
        if last_error:
            return f"Web search failed after retry: {last_error}"
        return f"No live web results found for '{query}'."

    snippets: List[str] = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        body = (r.get("body") or r.get("snippet") or "").strip()
        url = (r.get("href") or r.get("url") or "").strip()
        block = f"{i}. {title}\n   {body}"
        if url:
            block += f"\n   {url}"
        snippets.append(block)

    return f"Live web results for '{query}':\n\n" + "\n\n".join(snippets)


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_internal_database",
            "description": (
                "Search the company internal knowledge base for HR, policy, and "
                "engineering documents. Use this for questions about the user's uploaded "
                "policies, benefits, and processes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for internal documents.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_public_web",
            "description": (
                "Search the live public internet for current events, external facts, "
                "real-time prices, or news. Use this for anything that requires up-to-date "
                "market knowledge outside the internal database."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Web search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


SYSTEM_PROMPT = (
    "You are an autonomous agent solving real-world problems. Decide whether you need to use tools to answer each question.\n"
    "- Use 'search_internal_database' to read uploaded proprietary documents (like HR policies or internal knowledge).\n"
    "- Use 'search_public_web' for real-time market prices, current events, and live outside-world facts.\n"
    "- If the question is conversational (e.g. 'Hello'), answer directly without any tool.\n"
    "- You may chain multiple tool calls when the question requires it (e.g., comparing an internal budget to a live market price).\n"
    "- If a tool returns an error or no information, adjust your query and try again, or state the facts you found honestly.\n"
    "- Do not repeat the same tool call with the same arguments. If an observation is unhelpful, change strategy or answer with what you have."
)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[truncated, {len(text) - limit} chars omitted]"


def run_agent(
    user_query: str,
    history: List[Dict[str, Any]],
    llm_model: str,
    vector_store: VectorStore,
    embed_model: str,
    max_steps: int = MAX_AGENT_STEPS,
    on_step: Optional[Callable[[TraceStep], None]] = None,
) -> Tuple[str, List[TraceStep]]:
    trace: List[TraceStep] = []
    messages = list(history)
    messages.append({"role": "user", "content": user_query})

    def emit(step: TraceStep) -> None:
        trace.append(step)
        if on_step:
            on_step(step)

    for _ in range(max_steps):
        try:
            response = ollama.chat(
                model=llm_model,
                messages=messages,
                tools=TOOL_SPECS,
                options={"temperature": 0.1},
            )
        except ollama.ResponseError as exc:
            msg = f"Ollama error: {exc}. Ensure `ollama pull {llm_model}` succeeded."
            emit(TraceStep(kind="error", content=msg))
            return msg, trace
        except Exception as exc:
            msg = f"Agent call failed: {exc}"
            emit(TraceStep(kind="error", content=msg))
            return msg, trace

        message = _get(response, "message", {})
        content = (_get(message, "content") or "").strip()
        tool_calls = _get(message, "tool_calls") or []

        if content and not tool_calls:
            emit(TraceStep(kind="final", content=content))
            return content, trace

        if content:
            emit(TraceStep(kind="thought", content=content))

        if not tool_calls:
            fallback = content or "(agent returned no content)"
            emit(TraceStep(kind="final", content=fallback))
            return fallback, trace

        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "function": {
                        "name": _get(_get(tc, "function", {}), "name", ""),
                        "arguments": _get(_get(tc, "function", {}), "arguments", {}) or {},
                    }
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            fn = _get(tc, "function", {})
            name = _get(fn, "name", "")
            args = _get(fn, "arguments", {}) or {}

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"query": args}

            emit(TraceStep(kind="action", content=f"Call {name}", tool=name, args=args))

            start = time.perf_counter()
            try:
                q = str(args.get("query", "")).strip()
                if not q:
                    result = f"Tool '{name}' called with an empty query. Retry with a specific search term."
                elif name == "search_internal_database":
                    result = tool_search_internal(q, vector_store, embed_model)
                elif name == "search_public_web":
                    result = tool_search_web(q)
                else:
                    result = (
                        f"Unknown tool '{name}'. Available tools: "
                        "search_internal_database, search_public_web."
                    )
            except Exception as exc:
                result = f"Tool '{name}' failed: {exc}"
            elapsed = time.perf_counter() - start

            emit(TraceStep(
                kind="observation",
                content=str(result),
                tool=name,
                elapsed_s=elapsed,
            ))

            messages.append({
                "role": "tool",
                "content": _truncate(str(result), MAX_OBSERVATION_CHARS),
                "tool_call_id": _get(tc, "id", "call_0"),
                "name": name,
            })

    msg = "Agent reached the maximum step limit without producing a final answer."
    emit(TraceStep(kind="error", content=msg))
    return msg, trace


st.set_page_config(
    page_title="Live Agentic RAG",
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

    .trace-step {
        padding: 0.7rem 0.9rem;
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 0.5rem;
        background: var(--panel-2);
    }

    .trace-step.thought { border-left: 3px solid var(--muted); }
    .trace-step.action { border-left: 3px solid var(--accent); }
    .trace-step.observation { border-left: 3px solid var(--good); }
    .trace-step.final { border-left: 3px solid var(--accent-2); }
    .trace-step.error { border-left: 3px solid var(--bad); }

    .trace-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .trace-step.thought .trace-label { color: var(--muted); }
    .trace-step.action .trace-label { color: var(--accent-2); }
    .trace-step.observation .trace-label { color: var(--good); }
    .trace-step.final .trace-label { color: var(--accent-2); }
    .trace-step.error .trace-label { color: var(--bad); }

    .trace-body {
        color: var(--text);
        font-size: 0.85rem;
        line-height: 1.55;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .trace-mono {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.76rem;
        color: var(--muted);
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.5rem 0.6rem;
        margin-top: 0.35rem;
        overflow-x: auto;
        white-space: pre-wrap;
    }

    .trace-latency {
        float: right;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.7rem;
        color: var(--accent-2);
        background: rgba(91, 141, 239, 0.08);
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        border: 1px solid rgba(91, 141, 239, 0.2);
        text-transform: none;
        letter-spacing: 0;
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

    .stChatInput textarea {
        background: var(--panel-2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
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
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if "traces" not in st.session_state:
    st.session_state.traces: List[List[TraceStep]] = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

available_models = list_ollama_models()
ollama_ok = len(available_models) > 0
default_llm = pick_model(available_models, PREFERRED_LLM) or "llama3.2"

with st.sidebar:
    st.markdown("### Workspace")

    st.markdown("#### Models")
    if ollama_ok:
        llm_model = st.selectbox(
            "LLM model",
            available_models,
            index=available_models.index(default_llm) if default_llm in available_models else 0,
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
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.traces = []
        st.rerun()
    if st.button("Clear Index", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.indexed_docs = []
        st.rerun()

    st.markdown("#### Suggested Prompts")
    suggestions = [
        "What is the company hardware stipend limit?",
        "What is the live, real-world price of an iPhone 15 Pro right now?",
        "Can I afford to buy an iPhone 15 Pro using only my company hardware stipend?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"suggest_{s}"):
            st.session_state.pending_prompt = s
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
            <div class="app-title">Live Agentic RAG</div>
            <div class="app-subtitle">The agent intelligently searches internal PDFs and the live web to solve real problems</div>
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

        extraction_bar = st.progress(0.0, text="Extracting text...")
        for i, uf in enumerate(uploaded_files):
            try:
                raw = extract_text_from_pdf(uf)
                all_chunks.extend(chunk_text(raw, uf.name, chunk_size, chunk_overlap))
                indexed_names.append(uf.name)
            except Exception as exc:
                st.warning(f"Could not read {uf.name}: {exc}")
            extraction_bar.progress(
                (i + 1) / len(uploaded_files),
                text=f"Extracting {i + 1} of {len(uploaded_files)}...",
            )
        extraction_bar.empty()

        if all_chunks:
            embed_bar = st.progress(0.0, text="Generating embeddings...")

            def _cb(done: int, total: int) -> None:
                embed_bar.progress(done / total, text=f"Embedding {done} of {total}...")

            st.session_state.vector_store.clear()
            try:
                st.session_state.vector_store.add_chunks(all_chunks, embed_model, _cb)
            except Exception as exc:
                embed_bar.empty()
                st.error(f"Embedding failed: {exc}. Try `ollama pull {embed_model}`.")
                st.stop()
            embed_bar.empty()
            st.session_state.indexed_docs = indexed_names
            st.success(f"Indexed {len(all_chunks)} chunks from {len(indexed_names)} documents.")


def render_trace_step(step: TraceStep) -> None:
    kind = step.kind
    label_map = {
        "thought": "Thought",
        "action": "Action",
        "observation": "Observation",
        "final": "Final Answer",
        "error": "Error",
    }
    label = label_map.get(kind, kind.title())

    latency_html = ""
    if step.elapsed_s:
        latency_html = f'<span class="trace-latency">{step.elapsed_s:.2f}s</span>'

    if kind == "action":
        args_json = json.dumps(step.args or {}, indent=2)
        st.markdown(
            f"""
            <div class="trace-step action">
                <div class="trace-label">{html.escape(label)} &mdash; {html.escape(step.tool or '')}{latency_html}</div>
                <div class="trace-mono">{html.escape(args_json)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif kind == "observation":
        preview = step.content
        if len(preview) > MAX_TRACE_PREVIEW_CHARS:
            preview = preview[:MAX_TRACE_PREVIEW_CHARS] + "..."
        st.markdown(
            f"""
            <div class="trace-step observation">
                <div class="trace-label">{html.escape(label)} &mdash; {len(step.content)} chars{latency_html}</div>
                <div class="trace-body">{html.escape(preview)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="trace-step {html.escape(kind)}">
                <div class="trace-label">{html.escape(label)}{latency_html}</div>
                <div class="trace-body">{html.escape(step.content)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_trace(trace: List[TraceStep]) -> None:
    if not trace:
        return
    with st.expander(f"Agent Trace ({len(trace)} steps)", expanded=False):
        for step in trace:
            render_trace_step(step)


assistant_turn = 0
for msg in st.session_state.messages[1:]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            if assistant_turn < len(st.session_state.traces):
                render_trace(st.session_state.traces[assistant_turn])
        assistant_turn += 1


if not st.session_state.messages[1:]:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-title">Agent ready</div>
            <div class="empty-body">
                Upload PDFs from the sidebar, then ask a question. Try a compound query
                like "Can I afford an iPhone 15 Pro on my hardware stipend?" to see the
                agent chain internal and live web search in a single turn.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


prompt = st.chat_input("Ask the agent anything...")
if not prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        trace_container = st.container()

        def on_step(step: TraceStep) -> None:
            with trace_container:
                render_trace_step(step)

        with st.spinner("Agent thinking..."):
            final, trace = run_agent(
                user_query=prompt,
                history=st.session_state.messages[:-1],
                llm_model=llm_model,
                vector_store=st.session_state.vector_store,
                embed_model=embed_model,
                max_steps=MAX_AGENT_STEPS,
                on_step=on_step,
            )

        st.markdown(final)

    st.session_state.messages.append({"role": "assistant", "content": final})
    st.session_state.traces.append(trace)