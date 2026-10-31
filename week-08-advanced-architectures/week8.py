from __future__ import annotations

import html
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ollama
import streamlit as st

try:
    from pypdf import PdfReader
except ImportError:
    try:
        import PyPDF2
        PdfReader = PyPDF2.PdfReader
    except ImportError:
        PdfReader = None

try:
    from ddgs import DDGS
    _DDG_PKG = "ddgs"
except ImportError:
    try:
        from duckduckgo_search import DDGS
        _DDG_PKG = "duckduckgo-search"
    except ImportError:
        DDGS = None
        _DDG_PKG = None


DEFAULT_EMBED_MODEL = "nomic-embed-text"
PREFERRED_LLM = ["llama3.2", "llama3.1", "llama3", "mistral", "qwen2.5", "phi3"]
TOP_K = 4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_CONTEXT_CHARS = 8000
MAX_WEB_RESULTS = 3
WEB_RETRY_DELAY = 0.8
GRADES = ("CORRECT", "AMBIGUOUS", "INCORRECT")

SAMPLE_DOCUMENT_TITLE = "Acme Corp - Enterprise AI & Data Policy 2026.pdf"
SAMPLE_DOCUMENT_TEXT = """
Acme Corporation Enterprise AI Governance and Data Retention Policy (Revision 4.2, Q1 2026)

Section 1: Data Retention & Customer Interaction Logs
All prompts, inputs, and completions processed by internal LLM gateways are stored in encrypted cold storage for exactly 90 days. After 90 days, automated cryptographically verified purging routines permanently scrub conversation metadata and embeddings from all primary indices. Any audit requests regarding prompt interactions must be filed with the Legal Compliance Office within this 90-day retention window.

Section 2: Permitted Open-Source and Proprietary Model Architectures
Approved internal models for local deployment include Llama 3.2, Mistral-Small, and custom Qwen fine-tunes hosted within Acme's private on-premise Kubernetes clusters. Commercial frontier APIs (e.g. Claude, Gemini) may only be accessed via the centralized Enterprise API Broker with client-side Zero-Data-Retention (ZDR) agreements active. Direct unmonitored employee API keys to commercial AI providers are strictly prohibited.

Section 3: Fine-Tuning and Intellectual Property Guardrails
Customer proprietary codebases, confidential financial projections, and HR records are classified as Tier-4 Restricted Data and must NEVER be used for LoRA adaptation, model parameter fine-tuning, or vector indexing without explicit approval from the Chief Information Security Officer (CISO). External web search fallbacks in automated RAG pipelines must sanitize all internal hostnames and employee identifiers before dispatching queries to public search engines.
""".strip()


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def format_html_text(text: Any) -> str:
    """Safely escapes text for HTML and converts newlines to <br> to prevent CommonMark blank-line breakouts."""
    if not text:
        return ""
    return html.escape(str(text)).replace("\n", "<br>")


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
    if PdfReader is None:
        raise RuntimeError("PDF reader library (pypdf or PyPDF2) is not installed.")
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        try:
            extracted = page.extract_text() or ""
        except Exception:
            extracted = ""
        if extracted:
            text += extracted + "\n"
    return text


def chunk_text(
    text: str,
    source_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return []
    step = max(1, chunk_size - chunk_overlap)
    n = len(text)
    chunks: List[Dict[str, Any]] = []
    start = 0
    cid = 0
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
                "chunk_id": f"{source_name}::c{cid}",
                "source": source_name,
                "text": piece,
                "length": len(piece),
            })
            cid += 1
        if end >= n:
            break
        start += step
    return chunks


@st.cache_data(show_spinner=False, max_entries=10_000)
def get_embedding(text: str, model: str) -> List[float]:
    resp = ollama.embeddings(model=model, prompt=text)
    return [float(x) for x in (_get(resp, "embedding", []) or [])]


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

    def search(
        self, query: str, embed_model: str, top_k: int = TOP_K
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self.records:
            return []
        q_vec = get_embedding(query, embed_model)
        scored = [(item, cosine_similarity(q_vec, item["vector"])) for item in self.records]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def sources(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.records:
            src = r.get("source", "unknown")
            out[src] = out.get(src, 0) + 1
        return out


@dataclass
class TraceStep:
    step_num: int
    kind: str
    title: str
    headline: str
    human_narrative: str
    tag: Optional[str] = None
    badge_type: str = "info"  # "success", "warning", "danger", "info", "purple"
    latency_s: float = 0.0
    details: Optional[str] = None


@dataclass
class CRAGResult:
    query: str
    grade: str
    reasoning: str
    retrieved: List[Tuple[Dict[str, Any], float]] = field(default_factory=list)
    rewritten_query: Optional[str] = None
    web_results: str = ""
    raw_web_items: List[Dict[str, str]] = field(default_factory=list)
    final_context: str = ""
    answer: str = ""
    shadow_answer: Optional[str] = None
    route: str = ""
    total_latency_s: float = 0.0
    trace: List[TraceStep] = field(default_factory=list)


GRADER_PROMPT = """You are a strict retrieval evaluator for a RAG system.

User question:
{query}

Retrieved context:
{context}

Decide whether the retrieved context contains enough factual information
to answer the question.

Grade meanings:
- CORRECT   : context fully contains the answer.
- AMBIGUOUS : context is on-topic but incomplete; the answer is partial.
- INCORRECT : context does not address the question at all.

Reply in this exact format, nothing else:

GRADE: <CORRECT|AMBIGUOUS|INCORRECT>
REASON: <one short sentence>"""


REWRITE_PROMPT = """Rewrite the question into a short, specific search-engine
query that will return better external results. Return only the rewritten
query, no explanation, no quotes.

Original question: {query}"""


FINAL_PROMPT_INTERNAL = (
    "You are a helpful assistant. Answer the question using ONLY the internal "
    "context below. Cite the chunk ids. If the context does not answer the "
    "question, say so plainly.\n\nContext:\n{context}"
)

FINAL_PROMPT_WEB = (
    "You are a helpful assistant. The internal knowledge base did not contain "
    "the answer, so the system retrieved external web results. Answer the "
    "question using ONLY the web results below. Mention that the answer came "
    "from an external source.\n\nWeb results:\n{context}"
)

FINAL_PROMPT_FUSED = (
    "You are a helpful assistant. The internal knowledge base was incomplete, "
    "so the system combined internal context with external web results. Use "
    "both to answer the question. Clearly indicate which facts came from the "
    "internal documents and which came from the web.\n\n"
    "Internal context:\n{internal}\n\nWeb results:\n{web}"
)

SHADOW_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the context "
    "below, even if it does not fully address the question.\n\nContext:\n{context}"
)


def parse_grade(raw: str) -> Tuple[str, str]:
    text = raw.strip()
    grade = "AMBIGUOUS"
    reason = ""

    grade_match = re.search(r"GRADE\s*:\s*(CORRECT|AMBIGUOUS|INCORRECT)", text, re.IGNORECASE)
    if grade_match:
        grade = grade_match.group(1).upper()
    else:
        for g in GRADES:
            if re.search(rf"\b{g}\b", text, re.IGNORECASE):
                grade = g
                break

    reason_match = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip().split("\n")[0]

    return grade, reason


def grade_context(query: str, context: str, llm_model: str) -> Tuple[str, str]:
    if not context.strip():
        return "INCORRECT", "No internal chunks were retrieved."
    prompt = GRADER_PROMPT.format(query=query, context=context)
    try:
        resp = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        raw = _get(_get(resp, "message", {}), "content", "")
        return parse_grade(raw)
    except Exception as exc:
        return "AMBIGUOUS", f"Grader failed: {exc}"


def rewrite_query(query: str, llm_model: str) -> str:
    prompt = REWRITE_PROMPT.format(query=query)
    try:
        resp = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        out = _get(_get(resp, "message", {}), "content", "").strip()
        return out.strip("\"'`") or query
    except Exception:
        return query


def web_search(query: str) -> Tuple[str, List[Dict[str, str]]]:
    if DDGS is None:
        msg = (
            "External web search is unavailable. "
            f"Install `pip install {_DDG_PKG or 'ddgs'}` to enable it."
        )
        return msg, []

    last_err: Optional[str] = None
    results: List[Dict[str, Any]] = []
    for attempt in range(2):
        try:
            with DDGS() as client:
                results = list(client.text(query, max_results=MAX_WEB_RESULTS))
            if results:
                break
        except Exception as exc:
            last_err = str(exc)
        if attempt == 0:
            time.sleep(WEB_RETRY_DELAY)

    if not results:
        msg = f"No external results for '{query}'." + (f" ({last_err})" if last_err else "")
        return msg, []

    blocks = []
    raw_items: List[Dict[str, str]] = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        body = (r.get("body") or r.get("snippet") or "").strip()
        url = (r.get("href") or r.get("url") or r.get("link") or "").strip()
        raw_items.append({"title": title, "body": body, "url": url, "index": str(i)})
        block = f"{i}. {title}\n   {body}"
        if url:
            block += f"\n   {url}"
        blocks.append(block)

    formatted = f"External results for '{query}':\n\n" + "\n\n".join(blocks)
    return formatted, raw_items


def build_internal_context(retrieved: List[Tuple[Dict[str, Any], float]]) -> str:
    blocks = []
    used = 0
    for chunk, _score in retrieved:
        block = f"[{chunk['chunk_id']}]\n{chunk['text']}"
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def generate(prompt: str, llm_model: str) -> str:
    try:
        resp = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        return _get(_get(resp, "message", {}), "content", "").strip()
    except Exception as exc:
        return f"Generation failed: {exc}"


def run_crag(
    query: str,
    vector_store: VectorStore,
    embed_model: str,
    llm_model: str,
    show_shadow: bool = False,
    on_step: Optional[Any] = None,
) -> CRAGResult:
    result = CRAGResult(query=query, grade="AMBIGUOUS", reasoning="")
    t_start = time.perf_counter()
    step_num = 1

    def emit(step: TraceStep) -> None:
        result.trace.append(step)
        if on_step:
            on_step(step)

    # 1. Retrieve
    t0 = time.perf_counter()
    retrieved = vector_store.search(query, embed_model, top_k=TOP_K)
    internal_context = build_internal_context(retrieved)
    result.retrieved = retrieved
    top_score = retrieved[0][1] if retrieved else 0.0

    emit(TraceStep(
        step_num=step_num,
        kind="retrieve",
        title="Vector Knowledge Retrieval",
        headline=f"Scanned {vector_store.size} vector records · Pulled top {len(retrieved)} matches",
        human_narrative=(
            f"The system embedded your query with '{embed_model}' and computed cosine similarity "
            f"against all {vector_store.size} indexed document chunks. It selected the top {len(retrieved)} "
            f"most relevant passages (peak similarity score: {top_score:.3f})."
        ),
        tag=f"{len(retrieved)} Chunks",
        badge_type="info",
        latency_s=time.perf_counter() - t0,
        details=f"Retrieved chunk IDs: {', '.join(c[0]['chunk_id'] for c in retrieved)}\n\nContext preview:\n" + internal_context[:500] + ("..." if len(internal_context) > 500 else ""),
    ))
    step_num += 1

    # 2. Grade
    if not retrieved:
        result.grade = "INCORRECT"
        result.reasoning = "Internal database is empty or yielded no semantic matches."
        emit(TraceStep(
            step_num=step_num,
            kind="grade",
            title="Retrieval Quality Assessment",
            headline="Evaluator Verdict: INCORRECT (No context)",
            human_narrative="No document chunks were found. The evaluator marked retrieval as INCORRECT and triggered the external search fallback.",
            tag="INCORRECT",
            badge_type="danger",
            latency_s=0.0,
            details=result.reasoning,
        ))
    else:
        t0 = time.perf_counter()
        grade, reason = grade_context(query, internal_context, llm_model)
        result.grade = grade
        result.reasoning = reason

        if grade == "CORRECT":
            human_msg = (
                "The evaluator confirmed the internal document chunks contain complete, factual answers "
                "to your question. External search is bypassed to ensure maximum grounding and zero web noise."
            )
            b_type = "success"
        elif grade == "AMBIGUOUS":
            human_msg = (
                f"The evaluator determined that the retrieved context is on-topic but incomplete: {reason}. "
                "To provide an accurate answer, it triggered a hybrid fusion fallback to supplement internal facts with live web search."
            )
            b_type = "warning"
        else:
            human_msg = (
                f"The evaluator concluded that the internal documents do NOT contain facts relevant to this question: {reason}. "
                "To prevent silent hallucinations or false refusals, the system discarded the internal context and routed to full external search."
            )
            b_type = "danger"

        emit(TraceStep(
            step_num=step_num,
            kind="grade",
            title="Retrieval Quality Assessment",
            headline=f"Evaluator Verdict: {grade}",
            human_narrative=human_msg,
            tag=grade,
            badge_type=b_type,
            latency_s=time.perf_counter() - t0,
            details=f"Evaluator Rationale:\n{reason}\n\nEvaluator Prompt Context Inspected:\n{internal_context[:400]}...",
        ))
    step_num += 1

    # 3. Dynamic Route & Optional Web Fallback
    if result.grade == "CORRECT":
        result.route = "INTERNAL"
        result.final_context = internal_context
        final_prompt = FINAL_PROMPT_INTERNAL.format(context=internal_context)
        emit(TraceStep(
            step_num=step_num,
            kind="route",
            title="Dynamic Architecture Routing",
            headline="Routed to Strategy: INTERNAL ONLY",
            human_narrative="Confidence in internal evidence is high. Routing directly to internal document synthesis with strict chunk citations.",
            tag="INTERNAL",
            badge_type="success",
            latency_s=0.0,
            details="Grounded exclusively in indexed PDF chunks. Web fallback bypassed.",
        ))
        step_num += 1
    else:
        # Rewrite
        t0 = time.perf_counter()
        rewritten = rewrite_query(query, llm_model)
        result.rewritten_query = rewritten
        emit(TraceStep(
            step_num=step_num,
            kind="rewrite",
            title="Search Query Optimization",
            headline=f'Transformed query ➔ "{rewritten}"',
            human_narrative=(
                "Conversational questions often produce poor search engine results. The LLM stripped conversational "
                "framing and generated a keyword-dense search query optimized for external search engines."
            ),
            tag="Rewritten",
            badge_type="purple",
            latency_s=time.perf_counter() - t0,
            details=f"Original Question:\n{query}\n\nRewritten Web Query:\n{rewritten}",
        ))
        step_num += 1

        # Web Search
        t0 = time.perf_counter()
        web_text, raw_items = web_search(rewritten)
        result.web_results = web_text
        result.raw_web_items = raw_items
        emit(TraceStep(
            step_num=step_num,
            kind="web",
            title="Live Web Knowledge Discovery",
            headline=f"Retrieved {len(raw_items)} live results via search engine",
            human_narrative=(
                f"Dispatched the optimized search query to DuckDuckGo API and gathered {len(raw_items)} live web results "
                f"to bridge knowledge gaps with verified real-world facts."
            ),
            tag=f"{len(raw_items)} Web Results",
            badge_type="purple",
            latency_s=time.perf_counter() - t0,
            details=web_text[:800] + ("..." if len(web_text) > 800 else ""),
        ))
        step_num += 1

        if result.grade == "AMBIGUOUS":
            result.route = "FUSED"
            result.final_context = (
                f"INTERNAL CONTEXT:\n{internal_context}\n\nEXTERNAL WEB CONTEXT:\n{web_text}"
            )
            final_prompt = FINAL_PROMPT_FUSED.format(
                internal=internal_context, web=web_text
            )
            emit(TraceStep(
                step_num=step_num,
                kind="route",
                title="Dynamic Architecture Routing",
                headline="Routed to Strategy: FUSED HYBRID (Internal + Web)",
                human_narrative=(
                    "The system combined the verified internal document chunks with fresh external search snippets, "
                    "instructing the generator to clearly cite which facts originate from internal docs vs external web."
                ),
                tag="FUSED",
                badge_type="warning",
                latency_s=0.0,
                details="Fused Context:\nInternal chunks + DuckDuckGo search snippets combined into synthesis prompt.",
            ))
        else:
            result.route = "EXTERNAL"
            result.final_context = f"EXTERNAL WEB CONTEXT:\n{web_text}"
            final_prompt = FINAL_PROMPT_WEB.format(context=web_text)
            emit(TraceStep(
                step_num=step_num,
                kind="route",
                title="Dynamic Architecture Routing",
                headline="Routed to Strategy: EXTERNAL ONLY (Full Fallback)",
                human_narrative=(
                    "The system rejected the internal document context as off-topic to eliminate hallucination risks, "
                    "routing synthesis exclusively to verified external web search results."
                ),
                tag="EXTERNAL",
                badge_type="danger",
                latency_s=0.0,
                details="External Context:\nInternal context rejected. Using external search snippets exclusively.",
            ))
        step_num += 1

    # 4. Generate Answer
    t0 = time.perf_counter()
    result.answer = generate(final_prompt, llm_model) or "(No response was generated by the model.)"
    emit(TraceStep(
        step_num=step_num,
        kind="answer",
        title="Grounded Answer Synthesis",
        headline="Final answer synthesized with strict attribution",
        human_narrative=(
            f"The LLM ({llm_model}) synthesized the final response using only the validated and routed context. "
            f"Strict grounding rules were enforced to eliminate hallucinations."
        ),
        tag="Completed",
        badge_type="success",
        latency_s=time.perf_counter() - t0,
        details=f"Synthesized Prompt ({len(final_prompt)} chars):\n" + final_prompt[:400] + "...",
    ))
    step_num += 1

    # 5. Shadow Generation (Standard RAG comparison)
    if show_shadow and result.grade != "CORRECT" and internal_context:
        t0 = time.perf_counter()
        shadow_prompt = SHADOW_PROMPT.format(context=internal_context)
        result.shadow_answer = generate(shadow_prompt, llm_model) or "(No shadow answer)"
        emit(TraceStep(
            step_num=step_num,
            kind="shadow",
            title="Shadow Benchmark (Naive Standard RAG)",
            headline="Unguided baseline generated for comparative failure analysis",
            human_narrative=(
                "To demonstrate what a standard naive RAG system would have done, the model answered using only "
                "the inadequate internal context without CRAG's corrective evaluation. This highlights silent failures and hallucinations."
            ),
            tag="Naive Baseline",
            badge_type="warning",
            latency_s=time.perf_counter() - t0,
            details=f"Naive Prompt Output:\n{result.shadow_answer}",
        ))

    result.total_latency_s = time.perf_counter() - t_start
    return result


st.set_page_config(
    page_title="Corrective RAG (CRAG) Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #080a10;
        --bg-elevated: #0d111c;
        --surface: #111624;
        --surface-card: #151b2c;
        --surface-card-hover: #1b2339;
        --surface-sunken: #090c15;
        --border: rgba(255, 255, 255, 0.08);
        --border-subtle: rgba(255, 255, 255, 0.04);
        --border-strong: rgba(255, 255, 255, 0.16);
        
        --text: #f1f5f9;
        --text-secondary: #94a3b8;
        --muted: #64748b;
        --muted-dim: #475569;
        
        --primary: #3b82f6;
        --primary-glow: rgba(59, 130, 246, 0.25);
        --accent-cyan: #0ea5e9;
        --accent-cyan-soft: rgba(14, 165, 233, 0.12);
        
        --good: #10b981;
        --good-glow: rgba(16, 185, 129, 0.15);
        --good-border: rgba(16, 185, 129, 0.35);
        
        --warn: #f59e0b;
        --warn-glow: rgba(245, 158, 11, 0.15);
        --warn-border: rgba(245, 158, 11, 0.35);
        
        --bad: #f43f5e;
        --bad-glow: rgba(244, 63, 94, 0.15);
        --bad-border: rgba(244, 63, 94, 0.35);
        
        --purple: #8b5cf6;
        --purple-glow: rgba(139, 92, 246, 0.15);
        --purple-border: rgba(139, 92, 246, 0.35);
        
        --radius-sm: 8px;
        --radius: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        
        --shadow-sm: 0 2px 10px rgba(0, 0, 0, 0.4);
        --shadow: 0 10px 30px -8px rgba(0, 0, 0, 0.6);
        --shadow-glow: 0 0 25px rgba(59, 130, 246, 0.15);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background: radial-gradient(1400px 800px at 20% -10%, #151d32 0%, #080a10 65%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 1300px !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
        margin: 1.2rem 0 !important;
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 0.72rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: var(--accent-cyan) !important;
        margin: 1.2rem 0 0.5rem 0 !important;
    }

    /* Top Navigation Banner */
    .top-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 1.2rem 1.6rem;
        background: linear-gradient(135deg, rgba(21, 27, 44, 0.85) 0%, rgba(13, 17, 28, 0.85) 100%);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow);
    }
    .brand-section {
        display: flex;
        align-items: center;
        gap: 1.1rem;
    }
    .brand-shield {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 8px 24px -4px rgba(14, 165, 233, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    .brand-title {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-sub {
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-top: 0.2rem;
        font-weight: 500;
    }
    .badge-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .badge-status.online {
        background: var(--good-glow);
        border: 1px solid var(--good-border);
        color: var(--good);
    }
    .badge-status.offline {
        background: var(--bad-glow);
        border: 1px solid var(--bad-border);
        color: var(--bad);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: currentColor;
        box-shadow: 0 0 10px currentColor;
        animation: pulse 2s infinite ease-in-out;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Decision Architecture Diagram Bar */
    .arch-diagram-card {
        background: var(--surface-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.4rem;
        box-shadow: var(--shadow-sm);
    }
    .arch-diagram-title {
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .arch-flow-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .flow-node {
        flex: 1;
        min-width: 120px;
        background: var(--surface-sunken);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.65rem 0.85rem;
        text-align: center;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .flow-node.active-path {
        border-color: var(--primary);
        background: linear-gradient(180deg, rgba(59, 130, 246, 0.16) 0%, rgba(13, 17, 28, 0.85) 100%);
        box-shadow: 0 0 16px rgba(59, 130, 246, 0.3);
    }
    .flow-node.correct {
        border-color: var(--good);
        background: var(--good-glow);
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.25);
    }
    .flow-node.ambiguous {
        border-color: var(--warn);
        background: var(--warn-glow);
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.25);
    }
    .flow-node.incorrect {
        border-color: var(--bad);
        background: var(--bad-glow);
        box-shadow: 0 0 16px rgba(244, 63, 94, 0.25);
    }
    .flow-node-step {
        font-size: 0.64rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted);
        font-weight: 700;
    }
    .flow-node-name {
        font-size: 0.82rem;
        font-weight: 800;
        color: var(--text);
        margin-top: 0.2rem;
    }
    .flow-arrow {
        color: var(--muted-dim);
        font-size: 1rem;
        font-weight: 800;
        padding: 0 0.15rem;
    }

    /* Metric Strip */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.85rem;
        margin-bottom: 1.4rem;
    }
    @media (max-width: 768px) {
        .metric-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .metric-card {
        background: var(--surface-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.9rem 1.2rem;
        box-shadow: var(--shadow-sm);
    }
    .metric-label {
        font-size: 0.66rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--muted);
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .metric-value.good { color: var(--good); }
    .metric-value.warn { color: var(--warn); }
    .metric-value.bad { color: var(--bad); }
    .metric-value.cyan { color: var(--accent-cyan); }
    .metric-value.purple { color: var(--purple); }

    /* Grade Evaluation Banner */
    .eval-card {
        position: relative;
        padding: 1.3rem 1.6rem;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border);
        margin-bottom: 1.4rem;
        overflow: hidden;
    }
    .eval-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 5px;
    }
    .eval-card.CORRECT {
        background: linear-gradient(180deg, var(--good-glow) 0%, rgba(17, 22, 36, 0.95) 100%);
        border-color: var(--good-border);
    }
    .eval-card.CORRECT::before { background: var(--good); box-shadow: 0 0 14px var(--good); }
    .eval-card.AMBIGUOUS {
        background: linear-gradient(180deg, var(--warn-glow) 0%, rgba(17, 22, 36, 0.95) 100%);
        border-color: var(--warn-border);
    }
    .eval-card.AMBIGUOUS::before { background: var(--warn); box-shadow: 0 0 14px var(--warn); }
    .eval-card.INCORRECT {
        background: linear-gradient(180deg, var(--bad-glow) 0%, rgba(17, 22, 36, 0.95) 100%);
        border-color: var(--bad-border);
    }
    .eval-card.INCORRECT::before { background: var(--bad); box-shadow: 0 0 14px var(--bad); }

    .eval-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.75rem;
        flex-wrap: wrap;
    }
    .eval-grade-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .eval-card.CORRECT .eval-grade-pill { color: var(--good); }
    .eval-card.AMBIGUOUS .eval-grade-pill { color: var(--warn); }
    .eval-card.INCORRECT .eval-grade-pill { color: var(--bad); }

    .eval-route-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.35rem 0.75rem;
        border-radius: var(--radius-sm);
        background: var(--surface-sunken);
        border: 1px solid var(--border-strong);
        color: var(--text-secondary);
    }
    .eval-reasoning {
        color: var(--text-secondary);
        font-size: 0.94rem;
        line-height: 1.6;
    }

    /* Answer Cards */
    .answer-card {
        background: linear-gradient(180deg, #171e30 0%, #111524 100%);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: var(--radius-xl);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.45), var(--shadow-glow);
    }
    .answer-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid var(--border);
    }
    .answer-title-group {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .answer-title {
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent-cyan);
    }
    .answer-body {
        color: var(--text);
        font-size: 0.98rem;
        line-height: 1.8;
    }

    /* Shadow Comparison Card */
    .shadow-card {
        background: linear-gradient(180deg, rgba(245, 158, 11, 0.09) 0%, #121624 100%);
        border: 1px solid var(--warn-border);
        border-radius: var(--radius-xl);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.4rem;
        height: 100%;
        box-shadow: var(--shadow-sm);
    }
    .shadow-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid rgba(245, 158, 11, 0.2);
    }
    .shadow-title {
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--warn);
    }

    /* Human-Friendly Interactive Timeline Trace */
    .timeline-container {
        position: relative;
        padding-left: 2.2rem;
        margin: 1.2rem 0;
    }
    .timeline-container::before {
        content: "";
        position: absolute;
        left: 16px;
        top: 24px;
        bottom: 24px;
        width: 2px;
        background: linear-gradient(180deg, var(--accent-cyan) 0%, var(--primary) 50%, var(--good) 100%);
        opacity: 0.35;
    }
    .timeline-card {
        position: relative;
        background: var(--surface-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        transition: all 0.25s ease;
        box-shadow: var(--shadow-sm);
    }
    .timeline-card:hover {
        background: var(--surface-card-hover);
        border-color: var(--border-strong);
        transform: translateY(-1px);
    }
    .timeline-card::before {
        content: "";
        position: absolute;
        left: -26px;
        top: 22px;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: var(--bg);
        border: 3px solid var(--primary);
        box-shadow: 0 0 10px var(--primary);
        z-index: 2;
    }
    .timeline-card.success::before { border-color: var(--good); box-shadow: 0 0 10px var(--good); }
    .timeline-card.warning::before { border-color: var(--warn); box-shadow: 0 0 10px var(--warn); }
    .timeline-card.danger::before { border-color: var(--bad); box-shadow: 0 0 10px var(--bad); }
    .timeline-card.purple::before { border-color: var(--purple); box-shadow: 0 0 10px var(--purple); }

    .timeline-top-row {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }
    .timeline-title-area {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .step-counter {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 6px;
        background: var(--surface-sunken);
        border: 1px solid var(--border-strong);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--text-secondary);
    }
    .timeline-step-title {
        font-size: 0.88rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        color: var(--text);
    }
    .timeline-headline {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--accent-cyan);
        margin-top: 0.15rem;
    }
    .timeline-meta-group {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .timeline-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 0.2rem 0.55rem;
        border-radius: var(--radius-sm);
        background: var(--surface-sunken);
        border: 1px solid var(--border);
    }
    .timeline-badge.success { color: var(--good); border-color: var(--good-border); background: var(--good-glow); }
    .timeline-badge.warning { color: var(--warn); border-color: var(--warn-border); background: var(--warn-glow); }
    .timeline-badge.danger { color: var(--bad); border-color: var(--bad-border); background: var(--bad-glow); }
    .timeline-badge.purple { color: var(--purple); border-color: var(--purple-border); background: var(--purple-glow); }
    .timeline-badge.info { color: var(--accent-cyan); border-color: rgba(14, 165, 233, 0.35); background: var(--accent-cyan-soft); }

    .timeline-duration {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--muted);
    }
    .timeline-narrative {
        color: var(--text-secondary);
        font-size: 0.88rem;
        line-height: 1.6;
        margin-top: 0.4rem;
    }

    /* Collapsible Technical Details inside Timeline */
    details.timeline-details {
        margin-top: 0.75rem;
        border-top: 1px solid var(--border-subtle);
        padding-top: 0.6rem;
    }
    details.timeline-details summary {
        font-size: 0.76rem;
        font-weight: 700;
        color: var(--accent-cyan);
        cursor: pointer;
        user-select: none;
        outline: none;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    details.timeline-details summary:hover {
        color: var(--primary);
    }
    .timeline-details-body {
        margin-top: 0.6rem;
        background: var(--surface-sunken);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.75rem 0.9rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-secondary);
        line-height: 1.6;
        word-break: break-word;
        max-height: 250px;
        overflow-y: auto;
    }

    /* Retrieved Chunk Cards */
    .chunk-card {
        background: var(--surface-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
        transition: all 0.2s ease;
    }
    .chunk-card:hover {
        background: var(--surface-card-hover);
        border-color: var(--border-strong);
    }
    .chunk-meta-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.6rem;
    }
    .chunk-id-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        font-weight: 700;
        color: var(--accent-cyan);
        background: var(--accent-cyan-soft);
        padding: 0.2rem 0.6rem;
        border-radius: var(--radius-sm);
        border: 1px solid rgba(14, 165, 233, 0.28);
    }
    .chunk-sim-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: var(--radius-sm);
        background: var(--surface-sunken);
        border: 1px solid var(--border-strong);
    }
    .chunk-text-box {
        color: var(--text-secondary);
        font-size: 0.86rem;
        line-height: 1.65;
    }
    .sim-bar-bg {
        height: 5px;
        background: var(--surface-sunken);
        border-radius: 9999px;
        margin-top: 0.75rem;
        overflow: hidden;
    }
    .sim-bar-fill {
        height: 100%;
        border-radius: 9999px;
    }

    /* Web Result Cards */
    .web-card {
        background: var(--surface-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.2rem;
        margin-bottom: 0.85rem;
    }
    .web-card-title {
        font-size: 0.94rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.4rem;
    }
    .web-card-snippet {
        color: var(--text-secondary);
        font-size: 0.84rem;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    .web-card-link {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: var(--muted);
        text-decoration: none;
        word-break: break-all;
    }
    .web-card-link:hover {
        color: var(--accent-cyan);
        text-decoration: underline;
    }

    /* Preset Test Prompts */
    .preset-heading {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0.8rem 0 0.5rem 0;
    }

    /* Native Streamlit Element Overrides */
    .stTextInput input {
        background: var(--surface-sunken) !important;
        border: 1px solid var(--border-strong) !important;
        color: var(--text) !important;
        border-radius: var(--radius) !important;
        font-size: 0.96rem !important;
        padding: 0.8rem 1.1rem !important;
    }
    .stTextInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px var(--primary-glow) !important;
    }

    .stButton > button {
        border-radius: var(--radius) !important;
        font-weight: 700 !important;
        font-size: 0.86rem !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.6rem;
        border-bottom: 1px solid var(--border);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        color: var(--muted) !important;
        font-size: 0.84rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.1rem !important;
        border-radius: var(--radius) var(--radius) 0 0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-cyan) !important;
        background: var(--surface-card) !important;
        border-bottom: 2px solid var(--accent-cyan) !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Empty state card */
    .empty-card {
        text-align: center;
        padding: 3.8rem 2.2rem;
        border: 1px dashed var(--border-strong);
        border-radius: var(--radius-xl);
        background: rgba(17, 22, 36, 0.45);
        margin: 1.5rem 0;
    }
    .empty-icon {
        font-size: 2.8rem;
        margin-bottom: 0.9rem;
    }
    .empty-heading {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--text);
        margin-bottom: 0.5rem;
    }
    .empty-desc {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.65;
        max-width: 550px;
        margin: 0 auto;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "active_query" not in st.session_state:
    st.session_state.active_query = ""


available_models = list_ollama_models()
ollama_ok = len(available_models) > 0
default_llm = pick_model(available_models, PREFERRED_LLM) or "llama3.2"


# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Inference Models")
    if ollama_ok:
        llm_model = st.selectbox(
            "Local LLM",
            available_models,
            index=available_models.index(default_llm) if default_llm in available_models else 0,
            help="Select the local model running on your Ollama server for evaluation and synthesis.",
            key="select_llm_model",
        )
    else:
        llm_model = default_llm
        st.warning("No Ollama models detected. Please run `ollama serve`.")

    embed_model = st.text_input(
        "Embedding Model",
        value=DEFAULT_EMBED_MODEL,
        help="Ollama embedding model used for vector representations.",
        key="input_embed_model",
    )

    st.markdown("### 📂 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload enterprise documents or policies to index.",
        key="uploader_pdfs",
    )

    sample_clicked = st.button("📄 Load Built-in Demo Policy", use_container_width=True, key="btn_sidebar_sample")

    st.markdown("### ✂️ Chunking Bounds")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        chunk_size = st.number_input("Chunk Size", min_value=150, max_value=2000, value=CHUNK_SIZE, step=50, key="num_chunk_size")
    with col_c2:
        chunk_overlap = st.number_input("Overlap", min_value=0, max_value=500, value=CHUNK_OVERLAP, step=25, key="num_chunk_overlap")

    if chunk_overlap >= chunk_size:
        st.warning("Overlap clamped below chunk size.")
        chunk_overlap = max(0, chunk_size - 50)

    st.markdown("### 🛡️ Pipeline Guardrails")
    show_shadow = st.toggle(
        "Enable Shadow Benchmark",
        value=True,
        help="Runs an unguided naive RAG pipeline alongside CRAG when context is Ambiguous or Incorrect, highlighting silent failures vs. healed outputs.",
        key="toggle_shadow_benchmark",
    )

    st.markdown("### 🚀 Index Management")
    process_clicked = st.button("Process & Build Vector Index", type="primary", use_container_width=True, key="btn_process_index")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Clear Index", use_container_width=True, key="btn_clear_index"):
            st.session_state.vector_store.clear()
            st.session_state.indexed_docs = []
            st.session_state.last_result = None
            st.rerun()
    with col_btn2:
        if st.button("Clear Results", use_container_width=True, key="btn_clear_results"):
            st.session_state.last_result = None
            st.rerun()

    # Index Summary
    st.markdown("### 📊 Index Statistics")
    doc_count = len(st.session_state.indexed_docs)
    chunk_count = st.session_state.vector_store.size
    st.caption(f"**Indexed Documents:** {doc_count} document(s)")
    st.caption(f"**Total Vector Chunks:** {chunk_count} chunk(s)")

    if st.session_state.vector_store.sources:
        with st.expander("Inspect Indexed Sources"):
            for src, cnt in st.session_state.vector_store.sources.items():
                st.write(f"- `{src}`: {cnt} chunks")

    if not _DDG_PKG:
        st.error("DuckDuckGo package missing. Run: `pip install ddgs`")


# Quick Demo Loader Action
if sample_clicked:
    chunks = chunk_text(SAMPLE_DOCUMENT_TEXT, SAMPLE_DOCUMENT_TITLE, chunk_size, chunk_overlap)
    with st.spinner("Embedding built-in demo policy..."):
        st.session_state.vector_store.clear()
        st.session_state.vector_store.add_chunks(chunks, embed_model)
        st.session_state.indexed_docs = [SAMPLE_DOCUMENT_TITLE]
        st.session_state.last_result = None
    st.success(f"Loaded demo document: '{SAMPLE_DOCUMENT_TITLE}' with {len(chunks)} chunks.")
    st.rerun()


# Top Navigation / Hero Header
status_cls = "online" if ollama_ok else "offline"
status_txt = f"Ollama Online ({llm_model})" if ollama_ok else "Ollama Disconnected"

st.markdown(
    f'<div class="top-banner">'
    f'<div class="brand-section">'
    f'<div class="brand-shield">🛡️</div>'
    f'<div>'
    f'<h1 class="brand-title">Corrective RAG Studio</h1>'
    f'<p class="brand-sub">Self-healing retrieval architecture · Evaluator gating, automated search fallback, and grounded synthesis</p>'
    f'</div>'
    f'</div>'
    f'<div>'
    f'<span class="badge-status {status_cls}">'
    f'<span class="pulse-dot"></span>'
    f'{status_txt}'
    f'</span>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)


if not ollama_ok:
    st.error(
        "🚨 Cannot connect to Ollama service or no models are installed. "
        "Please launch `ollama serve` and pull a model such as `ollama pull llama3.2` and `ollama pull nomic-embed-text`."
    )
    st.stop()


# Document Ingestion Handler
if process_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file or click 'Load Built-in Demo Policy' in the sidebar.")
    else:
        all_chunks: List[Dict[str, Any]] = []
        indexed_names: List[str] = []

        extract_progress = st.progress(0.0, text="Extracting text from PDF files...")
        for i, uf in enumerate(uploaded_files):
            try:
                raw_text = extract_text_from_pdf(uf)
                if raw_text.strip():
                    doc_chunks = chunk_text(raw_text, uf.name, chunk_size, chunk_overlap)
                    all_chunks.extend(doc_chunks)
                    indexed_names.append(uf.name)
                else:
                    st.warning(f"File '{uf.name}' contained no extractable text.")
            except Exception as exc:
                st.error(f"Error reading '{uf.name}': {exc}")
            extract_progress.progress((i + 1) / len(uploaded_files))
        extract_progress.empty()

        if not all_chunks:
            st.error("No extractable text found in uploaded documents.")
        else:
            embed_progress = st.progress(0.0, text="Generating embeddings with Ollama...")

            def _cb(done: int, total: int) -> None:
                embed_progress.progress(done / total, text=f"Vectorizing chunk {done} of {total}...")

            st.session_state.vector_store.clear()
            try:
                st.session_state.vector_store.add_chunks(all_chunks, embed_model, _cb)
            except Exception as exc:
                embed_progress.empty()
                st.error(f"Embedding failed: {exc}. Please verify `ollama pull {embed_model}` is installed.")
                st.stop()

            embed_progress.empty()
            st.session_state.indexed_docs = indexed_names
            st.session_state.last_result = None
            st.success(f"Successfully indexed {len(all_chunks)} chunks across {len(indexed_names)} documents.")
            st.rerun()


has_index = st.session_state.vector_store.size > 0


def render_decision_flow(result: Optional[CRAGResult] = None):
    """Renders the CRAG architectural flow diagram with active path illumination."""
    active_grade = result.grade if result else ""
    active_route = result.route if result else ""

    node_c_cls = "correct active-path" if active_grade == "CORRECT" else ""
    node_a_cls = "ambiguous active-path" if active_grade == "AMBIGUOUS" else ""
    node_i_cls = "incorrect active-path" if active_grade == "INCORRECT" else ""

    route_int_cls = "active-path" if active_route == "INTERNAL" else ""
    route_fuse_cls = "active-path" if active_route == "FUSED" else ""
    route_ext_cls = "active-path" if active_route == "EXTERNAL" else ""

    html_code = (
        f'<div class="arch-diagram-card">'
        f'<div class="arch-diagram-title">'
        f'<span>Dynamic CRAG Architecture Pipeline</span>'
        f'<span>Active Route Strategy: <strong>{active_route or "STANDBY"}</strong></span>'
        f'</div>'
        f'<div class="arch-flow-row">'
        f'<div class="flow-node {"active-path" if result else ""}">'
        f'<div class="flow-node-step">01 / Input</div>'
        f'<div class="flow-node-name">User Query</div>'
        f'</div>'
        f'<div class="flow-arrow">→</div>'
        f'<div class="flow-node {"active-path" if result else ""}">'
        f'<div class="flow-node-step">02 / Search</div>'
        f'<div class="flow-node-name">Vector Retrieval</div>'
        f'</div>'
        f'<div class="flow-arrow">→</div>'
        f'<div class="flow-node {node_c_cls} {node_a_cls} {node_i_cls}">'
        f'<div class="flow-node-step">03 / Gate</div>'
        f'<div class="flow-node-name">LLM Evaluator</div>'
        f'</div>'
        f'<div class="flow-arrow">→</div>'
        f'<div class="flow-node {route_int_cls}">'
        f'<div class="flow-node-step">04 / Direct</div>'
        f'<div class="flow-node-name">Internal Grounded</div>'
        f'</div>'
        f'<div class="flow-arrow">/</div>'
        f'<div class="flow-node {route_fuse_cls}">'
        f'<div class="flow-node-step">04 / Hybrid</div>'
        f'<div class="flow-node-name">Web + Doc Fused</div>'
        f'</div>'
        f'<div class="flow-arrow">/</div>'
        f'<div class="flow-node {route_ext_cls}">'
        f'<div class="flow-node-step">04 / Fallback</div>'
        f'<div class="flow-node-name">External Web Only</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html_code, unsafe_allow_html=True)


def render_metrics_bar(result: CRAGResult):
    grade_cls = "good" if result.grade == "CORRECT" else "warn" if result.grade == "AMBIGUOUS" else "bad"
    route_cls = "good" if result.route == "INTERNAL" else "cyan" if result.route == "FUSED" else "purple"

    top_score = result.retrieved[0][1] if result.retrieved else 0.0

    html_code = (
        f'<div class="metric-grid">'
        f'<div class="metric-card">'
        f'<div class="metric-label">Evaluator Verdict</div>'
        f'<div class="metric-value {grade_cls}">{html.escape(result.grade)}</div>'
        f'</div>'
        f'<div class="metric-card">'
        f'<div class="metric-label">Routing Strategy</div>'
        f'<div class="metric-value {route_cls}">{html.escape(result.route)}</div>'
        f'</div>'
        f'<div class="metric-card">'
        f'<div class="metric-label">Passages / Top Match</div>'
        f'<div class="metric-value">{len(result.retrieved)} <span style="font-size:0.8rem; color:var(--muted)">({top_score:.3f})</span></div>'
        f'</div>'
        f'<div class="metric-card">'
        f'<div class="metric-label">Total Execution Time</div>'
        f'<div class="metric-value cyan">{result.total_latency_s:.2f}s</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html_code, unsafe_allow_html=True)


def render_evaluator_card(result: CRAGResult):
    route_desc = {
        "INTERNAL": "Factually Grounded · Web Search Bypassed · Direct Synthesis",
        "FUSED": "Partial Coverage · Query Rewritten · Internal + Web Fused",
        "EXTERNAL": "Irrelevant / Missing Internal Data · Full Web Search Fallback",
    }.get(result.route, result.route)

    icon = "✅" if result.grade == "CORRECT" else "⚠️" if result.grade == "AMBIGUOUS" else "🚨"
    reasoning_fmt = format_html_text(result.reasoning or "No detailed rationale provided by evaluator.")

    html_code = (
        f'<div class="eval-card {html.escape(result.grade)}">'
        f'<div class="eval-header">'
        f'<div class="eval-grade-pill">'
        f'<span>{icon}</span>'
        f'<span>Evaluation Gate: {html.escape(result.grade)}</span>'
        f'</div>'
        f'<div class="eval-route-badge">{html.escape(route_desc)}</div>'
        f'</div>'
        f'<div class="eval-reasoning">'
        f'<strong>Evaluator Rationale:</strong> {reasoning_fmt}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html_code, unsafe_allow_html=True)


def render_human_trace_timeline(trace: List[TraceStep], container=None):
    """Renders the human-friendly interactive investigation timeline cleanly without markdown code block breakouts."""
    cards_html = []
    for step in trace:
        dur_str = f"{step.latency_s:.2f}s" if step.latency_s > 0 else "instant"
        tag_badge = (
            f'<span class="timeline-badge {html.escape(step.badge_type)}">{html.escape(step.tag)}</span>'
            if step.tag else ""
        )

        details_block = ""
        if step.details:
            details_fmt = format_html_text(step.details)
            details_block = (
                f'<details class="timeline-details">'
                f'<summary>🔍 Inspect Technical Evidence &amp; Prompt Data</summary>'
                f'<div class="timeline-details-body">{details_fmt}</div>'
                f'</details>'
            )

        narrative_fmt = format_html_text(step.human_narrative)

        card = (
            f'<div class="timeline-card {html.escape(step.badge_type)}">'
            f'<div class="timeline-top-row">'
            f'<div class="timeline-title-area">'
            f'<span class="step-counter">{step.step_num:02d}</span>'
            f'<div>'
            f'<div class="timeline-step-title">{html.escape(step.title)}</div>'
            f'<div class="timeline-headline">{html.escape(step.headline)}</div>'
            f'</div>'
            f'</div>'
            f'<div class="timeline-meta-group">'
            f'{tag_badge}'
            f'<span class="timeline-duration">⏱️ {dur_str}</span>'
            f'</div>'
            f'</div>'
            f'<div class="timeline-narrative">{narrative_fmt}</div>'
            f'{details_block}'
            f'</div>'
        )
        cards_html.append(card)

    full_html = f'<div class="timeline-container">{"".join(cards_html)}</div>'
    if container is not None:
        container.markdown(full_html, unsafe_allow_html=True)
    else:
        st.markdown(full_html, unsafe_allow_html=True)


if has_index:
    render_decision_flow(st.session_state.last_result)

    # Preset Query Selector
    st.markdown('<div class="preset-heading">💡 Evaluation Presets (Click to Test Decision Gateways):</div>', unsafe_allow_html=True)
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("🎯 In-Scope: Data Retention Policy", use_container_width=True, help="Target: CORRECT · Grounded in internal chunks directly", key="btn_preset_correct"):
            st.session_state.active_query = "What is the exact data retention period for AI interactions and logs?"
            st.rerun()
    with col_p2:
        if st.button("🔀 Partial Match: Permitted Models", use_container_width=True, help="Target: AMBIGUOUS · Fuses internal policy with live web search", key="btn_preset_ambiguous"):
            st.session_state.active_query = "What models are permitted internally and what are the best practices for Llama 3.2 deployment?"
            st.rerun()
    with col_p3:
        if st.button("🌐 Out-of-Scope: Apple Market Cap", use_container_width=True, help="Target: INCORRECT · Rejects internal docs, executes web search fallback", key="btn_preset_incorrect"):
            st.session_state.active_query = "What is the current market cap of Apple and latest tech earnings news?"
            st.rerun()

    # Query Input Form with Enter key support
    with st.form("crag_query_form", clear_on_submit=False):
        user_query = st.text_input(
            "Enter your query",
            value=st.session_state.active_query,
            placeholder="Ask a question about the document or an external topic...",
            label_visibility="collapsed",
        )
        col_submit, col_hint = st.columns([1, 4])
        with col_submit:
            submitted = st.form_submit_button("Run CRAG Pipeline", type="primary", use_container_width=True)
        with col_hint:
            st.caption("Press **Enter** or click to evaluate retrieval, assess sufficiency, and execute dynamic routing.")

    if submitted and user_query.strip():
        st.session_state.active_query = user_query
        st.session_state.last_result = None

        trace_slot = st.empty()
        live_steps: List[TraceStep] = []

        def _step_callback(step: TraceStep):
            live_steps.append(step)
            render_human_trace_timeline(live_steps, trace_slot)

        with st.spinner("Executing Corrective RAG reasoning loop..."):
            res = run_crag(
                query=user_query,
                vector_store=st.session_state.vector_store,
                embed_model=embed_model,
                llm_model=llm_model,
                show_shadow=show_shadow,
                on_step=_step_callback,
            )

        st.session_state.last_result = res
        trace_slot.empty()
        st.rerun()

    # Display Results View
    if st.session_state.last_result:
        res: CRAGResult = st.session_state.last_result

        render_metrics_bar(res)
        render_evaluator_card(res)

        # Primary Answers Section (Side-by-side if Shadow Mode triggered)
        if res.shadow_answer:
            col_crag, col_shadow = st.columns(2)
            with col_crag:
                crag_ans_fmt = format_html_text(res.answer)
                st.markdown(
                    f'<div class="answer-card">'
                    f'<div class="answer-header">'
                    f'<div class="answer-title-group">'
                    f'<span style="font-size:1.1rem">🛡️</span>'
                    f'<span class="answer-title">CRAG Self-Healed Answer</span>'
                    f'</div>'
                    f'<span class="eval-route-badge">Route: {html.escape(res.route)}</span>'
                    f'</div>'
                    f'<div class="answer-body">{crag_ans_fmt}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_shadow:
                shadow_ans_fmt = format_html_text(res.shadow_answer)
                st.markdown(
                    f'<div class="shadow-card">'
                    f'<div class="shadow-header">'
                    f'<div class="answer-title-group">'
                    f'<span style="font-size:1.1rem">⚠️</span>'
                    f'<span class="shadow-title">Standard RAG (Silent Failure Baseline)</span>'
                    f'</div>'
                    f'<span class="eval-route-badge" style="color:var(--warn)">Unguided Naive</span>'
                    f'</div>'
                    f'<div class="answer-body" style="color:var(--text-secondary)">{shadow_ans_fmt}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            ans_fmt = format_html_text(res.answer)
            st.markdown(
                f'<div class="answer-card">'
                f'<div class="answer-header">'
                f'<div class="answer-title-group">'
                f'<span style="font-size:1.1rem">🛡️</span>'
                f'<span class="answer-title">CRAG Synthesized Response</span>'
                f'</div>'
                f'<span class="eval-route-badge">Route: {html.escape(res.route)}</span>'
                f'</div>'
                f'<div class="answer-body">{ans_fmt}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Multi-tab Deep Dive Workspace
        tab_trace, tab_chunks, tab_rewrite, tab_context, tab_json = st.tabs([
            "⏱️ Execution Narrative & Trace",
            "🧩 Retrieved Passages",
            "🌐 Query Rewrite & Search",
            "📄 Grounded Context",
            "🔍 JSON Audit Export",
        ])

        with tab_trace:
            st.markdown("**Step-by-Step CRAG Execution Narrative:**")
            render_human_trace_timeline(res.trace)

        with tab_chunks:
            if res.retrieved:
                st.markdown(f"**Top {len(res.retrieved)} Cosine Similarity Matches:**")
                for chunk, score in res.retrieved:
                    pct = max(0.0, min(100.0, score * 100))
                    bar_color = "var(--good)" if score >= 0.7 else "var(--primary)" if score >= 0.4 else "var(--muted)"
                    chunk_text_fmt = format_html_text(chunk.get('text', ''))
                    
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<div class="chunk-meta-row">'
                        f'<span class="chunk-id-tag">{html.escape(chunk.get("chunk_id", "chunk"))}</span>'
                        f'<span class="chunk-sim-pill" style="color:{bar_color}">Score: {score:.4f}</span>'
                        f'</div>'
                        f'<div class="chunk-text-box">{chunk_text_fmt}</div>'
                        f'<div class="sim-bar-bg">'
                        f'<div class="sim-bar-fill" style="width:{pct:.1f}%; background:{bar_color};"></div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No internal chunks retrieved for this query.")

        with tab_rewrite:
            col_rw1, col_rw2 = st.columns(2)
            with col_rw1:
                st.markdown("**Original User Query:**")
                st.info(res.query)
            with col_rw2:
                st.markdown("**Search-Engine Rewritten Query:**")
                if res.rewritten_query:
                    st.success(f"`{res.rewritten_query}`")
                else:
                    st.caption("No rewrite needed (internal grade was CORRECT).")

            if res.raw_web_items:
                st.markdown(f"**External Search Results ({len(res.raw_web_items)} items):**")
                for item in res.raw_web_items:
                    url = item.get("url", "")
                    link_html = f'<a href="{html.escape(url)}" target="_blank" class="web-card-link">🔗 {html.escape(url)}</a>' if url else ""
                    snippet_fmt = format_html_text(item.get('body', ''))
                    st.markdown(
                        f'<div class="web-card">'
                        f'<div class="web-card-title">{html.escape(item.get("title", "Web Result"))}</div>'
                        f'<div class="web-card-snippet">{snippet_fmt}</div>'
                        f'{link_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            elif res.web_results:
                st.text_area("External Results Text", res.web_results, height=200)

        with tab_context:
            st.markdown("**Full Grounded Context Injected into Prompt:**")
            st.text_area("Synthesizer Context", res.final_context or "(Empty context)", height=280)

        with tab_json:
            audit_dict = {
                "query": res.query,
                "grade": res.grade,
                "route": res.route,
                "reasoning": res.reasoning,
                "rewritten_query": res.rewritten_query,
                "total_latency_seconds": round(res.total_latency_s, 4),
                "chunks_retrieved_count": len(res.retrieved),
                "retrieved_chunks": [
                    {"chunk_id": c[0]["chunk_id"], "score": round(c[1], 4), "text": c[0]["text"][:150] + "..."}
                    for c in res.retrieved
                ],
                "trace": [asdict(t) for t in res.trace],
            }
            st.json(audit_dict)
            st.download_button(
                "📥 Download Audit Trail (.json)",
                data=json.dumps(audit_dict, indent=2),
                file_name="crag_audit_trace.json",
                mime="application/json",
                key="btn_download_audit_json",
            )

else:
    # Empty State Display
    st.markdown(
        '<div class="empty-card">'
        '<div class="empty-icon">🛡️</div>'
        '<div class="empty-heading">Corrective RAG (CRAG) Studio Active</div>'
        '<div class="empty-desc">'
        'Evaluate retrieval sufficiency before generating. If documents lack coverage, '
        'the pipeline rewrites the query and falls back to live web search to eliminate hallucinations. '
        'Upload PDF documents using the sidebar, or click below to test immediately with the Acme Corp policy.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_demo_c, _ = st.columns([2, 3])
    with col_demo_c:
        if st.button("⚡ Click Here to Instantly Load Demo Document", type="primary", use_container_width=True, key="btn_load_demo_empty"):
            chunks = chunk_text(SAMPLE_DOCUMENT_TEXT, SAMPLE_DOCUMENT_TITLE, chunk_size, chunk_overlap)
            with st.spinner("Indexing demo policy..."):
                st.session_state.vector_store.clear()
                st.session_state.vector_store.add_chunks(chunks, embed_model)
                st.session_state.indexed_docs = [SAMPLE_DOCUMENT_TITLE]
                st.session_state.last_result = None
            st.rerun()
