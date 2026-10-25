from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ollama
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader


DEFAULT_LLM = os.getenv("RAG_LLM", "llama3.2")
DEFAULT_EMBED_MODEL = os.getenv("RAG_EMBED", "nomic-embed-text")
TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

CACHE_THRESHOLD = 0.85
CACHE_MAX_SIZE = 500
CACHE_TTL_SECONDS = 3600
CACHE_NEAR_MISS_MARGIN = 0.05

MAX_CONTEXT_CHARS = 8000


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


_client: Optional[ollama.AsyncClient] = None


def get_client() -> ollama.AsyncClient:
    global _client
    if _client is None:
        _client = ollama.AsyncClient()
    return _client


async def list_models() -> List[str]:
    try:
        resp = await get_client().list()
        models = _get(resp, "models", []) or []
        return sorted(n for m in models if (n := _get(m, "model") or _get(m, "name")))
    except Exception:
        return []


async def get_embedding(text: str, model: str = DEFAULT_EMBED_MODEL) -> List[float]:
    resp = await get_client().embeddings(model=model, prompt=text)
    vec = _get(resp, "embedding", []) or []
    return [float(x) for x in vec]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return []
    step = max(1, chunk_size - overlap)
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
            chunks.append({"chunk_id": f"{source}::{cid}", "source": source, "text": piece})
            cid += 1
        if end >= n:
            break
        start += step
    return chunks


class VectorStore:
    def __init__(self) -> None:
        self.chunks: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None
        self._dirty = True

    def add(self, chunks: List[Dict[str, Any]], vectors: List[List[float]]) -> None:
        for chunk, vec in zip(chunks, vectors):
            chunk["vector"] = np.asarray(vec, dtype=np.float32)
            self.chunks.append(chunk)
        self._dirty = True

    def _build(self) -> None:
        if not self._dirty:
            return
        if not self.chunks:
            self._matrix = None
            self._dirty = False
            return
        mat = np.vstack([c["vector"] for c in self.chunks])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self._matrix = mat / norms
        self._dirty = False

    def retrieve(self, query_vec: List[float], top_k: int = TOP_K) -> List[Tuple[Dict[str, Any], float]]:
        self._build()
        if self._matrix is None:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-10)
        sims = self._matrix @ q
        k = min(top_k, len(self.chunks))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.chunks[i], float(sims[i])) for i in idx]

    def clear(self) -> None:
        self.chunks = []
        self._matrix = None
        self._dirty = True

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self.chunks:
            out[c["source"]] = out.get(c["source"], 0) + 1
        return out


vector_store = VectorStore()


@dataclass
class CacheEntry:
    query: str
    vector: np.ndarray
    answer: str
    created_at: float
    last_access: float
    hits: int = 0
    original_generation_ms: float = 0.0


class SemanticCache:
    def __init__(
        self,
        threshold: float = CACHE_THRESHOLD,
        max_size: int = CACHE_MAX_SIZE,
        ttl: float = CACHE_TTL_SECONDS,
    ) -> None:
        self.threshold = threshold
        self.max_size = max_size
        self.ttl = ttl
        self._entries: Dict[str, CacheEntry] = {}
        self._matrix: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._dirty = True
        self.hits = 0
        self.misses = 0
        self.near_misses = 0
        self.total_saved_ms = 0.0
        self.near_miss_log: List[Dict[str, Any]] = []
        self._log_limit = 50

    def _ensure(self) -> None:
        if not self._dirty:
            return
        if not self._entries:
            self._matrix = None
            self._ids = []
            self._dirty = False
            return
        self._ids = list(self._entries.keys())
        mat = np.vstack([self._entries[k].vector for k in self._ids])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self._matrix = mat / norms
        self._dirty = False

    def _prune_expired(self) -> None:
        now = time.time()
        expired = [k for k, e in self._entries.items() if now - e.created_at > self.ttl]
        for k in expired:
            del self._entries[k]
        if expired:
            self._dirty = True

    @staticmethod
    def _normalize(vec: List[float]) -> np.ndarray:
        q = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(q)
        return q / n if n else q

    def nearest(self, query_vec: List[float]) -> Optional[Dict[str, Any]]:
        self._prune_expired()
        self._ensure()
        if self._matrix is None or not self._ids:
            return None
        q = self._normalize(query_vec)
        sims = self._matrix @ q
        best = int(np.argmax(sims))
        score = float(sims[best])
        entry = self._entries[self._ids[best]]
        return {
            "similarity": round(score, 4),
            "cached_query": entry.query,
            "answer_preview": entry.answer[:120] + ("..." if len(entry.answer) > 120 else ""),
            "hits": entry.hits,
        }

    def check(self, query_vec: List[float]) -> Optional[Dict[str, Any]]:
        self._prune_expired()
        self._ensure()
        if self._matrix is None or not self._ids:
            self.misses += 1
            return None
        q = self._normalize(query_vec)
        sims = self._matrix @ q
        best = int(np.argmax(sims))
        score = float(sims[best])
        entry = self._entries[self._ids[best]]

        if score >= self.threshold:
            entry.hits += 1
            entry.last_access = time.time()
            self.hits += 1
            self.total_saved_ms += entry.original_generation_ms
            return {
                "answer": entry.answer,
                "similarity": score,
                "cached_query": entry.query,
                "saved_ms": entry.original_generation_ms,
            }

        self.misses += 1
        if score >= self.threshold - CACHE_NEAR_MISS_MARGIN:
            self.near_misses += 1
            self.near_miss_log.append({
                "ts": time.time(),
                "similarity": round(score, 4),
                "threshold": self.threshold,
                "cached_query": entry.query,
            })
            if len(self.near_miss_log) > self._log_limit:
                self.near_miss_log.pop(0)
        return None

    def add(self, query: str, query_vec: List[float], answer: str, generation_ms: float) -> None:
        if query in self._entries:
            del self._entries[query]
        while len(self._entries) >= self.max_size:
            oldest = min(self._entries, key=lambda k: self._entries[k].last_access)
            del self._entries[oldest]
        now = time.time()
        self._entries[query] = CacheEntry(
            query=query,
            vector=np.asarray(query_vec, dtype=np.float32),
            answer=answer,
            created_at=now,
            last_access=now,
            original_generation_ms=generation_ms,
        )
        self._dirty = True

    def set_threshold(self, threshold: float) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = float(threshold)

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._entries),
            "max_size": self.max_size,
            "threshold": round(self.threshold, 4),
            "hits": self.hits,
            "misses": self.misses,
            "near_misses": self.near_misses,
            "total_requests": total,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "total_saved_ms": round(self.total_saved_ms, 1),
        }

    def clear(self) -> None:
        self._entries.clear()
        self._matrix = None
        self._ids = []
        self._dirty = True
        self.hits = 0
        self.misses = 0
        self.near_misses = 0
        self.total_saved_ms = 0.0
        self.near_miss_log.clear()


semantic_cache = SemanticCache()


class HealthResponse(BaseModel):
    status: str
    ollama_ok: bool
    models: List[str]
    indexed_documents: int
    indexed_chunks: int
    llm_model: str
    embed_model: str


class UploadResponse(BaseModel):
    status: str
    filename: str
    pages: int
    chunks: int
    total_chunks: int


class CacheStatsResponse(BaseModel):
    size: int
    max_size: int
    threshold: float
    hits: int
    misses: int
    near_misses: int
    total_requests: int
    hit_rate: float
    total_saved_ms: float


class ThresholdPayload(BaseModel):
    threshold: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    models = await list_models()
    if not models:
        print("WARNING: No Ollama models detected.")
        print("         Start `ollama serve` and pull a model, e.g. `ollama pull llama3.2`.")
    else:
        print(f"Ollama ready. {len(models)} model(s) available.")
    print(f"Semantic cache threshold: {semantic_cache.threshold}")
    yield


app = FastAPI(title="Production RAG API (Week 7)", version="1.2", lifespan=lifespan)


def sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    models = await list_models()
    return HealthResponse(
        status="ok",
        ollama_ok=len(models) > 0,
        models=models,
        indexed_documents=len(vector_store.sources),
        indexed_chunks=vector_store.size,
        llm_model=DEFAULT_LLM,
        embed_model=DEFAULT_EMBED_MODEL,
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats() -> CacheStatsResponse:
    return CacheStatsResponse(**semantic_cache.stats())


@app.post("/cache/threshold")
async def cache_set_threshold(payload: ThresholdPayload) -> Dict[str, Any]:
    try:
        semantic_cache.set_threshold(payload.threshold)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"threshold": semantic_cache.threshold}


@app.get("/cache/nearest")
async def cache_nearest(query: str = Query(..., min_length=1, max_length=2000)) -> Dict[str, Any]:
    if not semantic_cache._entries:
        return {"query": query, "nearest": None, "threshold": semantic_cache.threshold}
    try:
        qv = await get_embedding(query)
    except Exception as exc:
        raise HTTPException(500, f"embedding failed: {exc}")
    nearest = semantic_cache.nearest(qv)
    return {
        "query": query,
        "nearest": nearest,
        "threshold": semantic_cache.threshold,
        "would_hit": bool(nearest and nearest["similarity"] >= semantic_cache.threshold),
    }


@app.get("/cache/near-misses")
async def cache_near_misses() -> Dict[str, Any]:
    return {
        "threshold": semantic_cache.threshold,
        "near_miss_margin": CACHE_NEAR_MISS_MARGIN,
        "count": len(semantic_cache.near_miss_log),
        "entries": list(reversed(semantic_cache.near_miss_log)),
    }


@app.post("/cache/clear")
async def cache_clear() -> Dict[str, str]:
    semantic_cache.clear()
    return {"status": "cleared"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    try:
        reader = PdfReader(file.file)
        pages_text: List[str] = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                pages_text.append(text)
    except Exception as exc:
        raise HTTPException(400, f"Could not parse PDF: {exc}")
    if not pages_text:
        raise HTTPException(400, "No extractable text found")

    full_text = "\n".join(pages_text)
    chunks = chunk_text(full_text, file.filename)
    if not chunks:
        raise HTTPException(400, "No chunks produced")

    try:
        semaphore = asyncio.Semaphore(4)

        async def embed_one(chunk: Dict[str, Any]) -> List[float]:
            async with semaphore:
                return await get_embedding(chunk["text"])

        vectors = await asyncio.gather(*[embed_one(c) for c in chunks])
    except Exception as exc:
        raise HTTPException(500, f"Embedding failed: {exc}")

    vector_store.add(chunks, vectors)
    return UploadResponse(
        status="success",
        filename=file.filename,
        pages=len(pages_text),
        chunks=len(chunks),
        total_chunks=vector_store.size,
    )


@app.get("/documents")
async def list_documents() -> Dict[str, Any]:
    return {
        "documents": [{"source": s, "chunks": c} for s, c in vector_store.sources.items()],
        "total_chunks": vector_store.size,
    }


@app.post("/documents/clear")
async def clear_documents() -> Dict[str, str]:
    vector_store.clear()
    return {"status": "cleared"}


@app.get("/chat/stream")
async def chat_stream(query: str = Query(..., min_length=1, max_length=2000)):
    if not vector_store.chunks:
        async def no_docs():
            yield sse_event("meta", {"type": "error", "message": "No documents indexed."})
            yield sse_event("done", {"total_ms": 0, "tokens": 0, "cached": False})
        return StreamingResponse(no_docs(), media_type="text/event-stream")

    try:
        query_vec = await get_embedding(query)
    except Exception as exc:
        async def embed_err():
            yield sse_event("meta", {"type": "error", "message": f"Embedding failed: {exc}"})
            yield sse_event("done", {"total_ms": 0, "tokens": 0, "cached": False})
        return StreamingResponse(embed_err(), media_type="text/event-stream")

    nearest = semantic_cache.nearest(query_vec)
    cached = semantic_cache.check(query_vec)

    if cached:
        return StreamingResponse(stream_cache_hit(cached), media_type="text/event-stream")

    results = vector_store.retrieve(query_vec, top_k=TOP_K)
    blocks: List[str] = []
    used = 0
    for chunk, _score in results:
        block = f"[{chunk['chunk_id']}]\n{chunk['text']}"
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        used += len(block)
    context = "\n\n".join(blocks)

    return StreamingResponse(
        stream_fresh(query, query_vec, context, nearest),
        media_type="text/event-stream",
    )


async def stream_cache_hit(cached: Dict[str, Any]):
    start = time.perf_counter()
    yield sse_event("meta", {
        "type": "cache_hit",
        "similarity": round(cached["similarity"], 4),
        "cached_query": cached["cached_query"],
        "saved_ms": round(cached["saved_ms"], 1),
        "threshold": semantic_cache.threshold,
    })
    yield sse_event("token", {"text": cached["answer"]})
    total_ms = (time.perf_counter() - start) * 1000
    yield sse_event("done", {
        "ttft_ms": round(total_ms, 2),
        "total_ms": round(total_ms, 2),
        "tokens": len(cached["answer"].split()),
        "cached": True,
        "similarity": round(cached["similarity"], 4),
    })


async def stream_fresh(
    query: str,
    query_vec: List[float],
    context: str,
    nearest: Optional[Dict[str, Any]] = None,
):
    start = time.perf_counter()
    first_token_at: Optional[float] = None
    full = ""
    token_count = 0

    meta: Dict[str, Any] = {
        "type": "fresh",
        "context_chars": len(context),
        "threshold": round(semantic_cache.threshold, 4),
    }
    if nearest:
        meta["nearest"] = nearest
        meta["nearest_gap"] = round(semantic_cache.threshold - nearest["similarity"], 4)
    yield sse_event("meta", meta)

    system = (
        "You are an expert assistant. Answer using ONLY the context below. "
        "Always cite sources with the chunk id, e.g. [file.pdf::3]. "
        "If the answer is not in the context, say 'I cannot answer this based on the provided document.'"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    try:
        client = get_client()
        async for part in await client.chat(model=DEFAULT_LLM, messages=messages, stream=True):
            content = _get(_get(part, "message", {}), "content", "")
            if not content:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
                yield sse_event("ttft", {"ttft_ms": round((first_token_at - start) * 1000, 1)})
            full += content
            token_count += 1
            yield sse_event("token", {"text": content})
    except Exception as exc:
        yield sse_event("stream_error", {"message": str(exc)})
        return

    total_ms = (time.perf_counter() - start) * 1000
    if full.strip():
        semantic_cache.add(query, query_vec, full, total_ms)

    yield sse_event("done", {
        "ttft_ms": round((first_token_at - start) * 1000, 1) if first_token_at else None,
        "total_ms": round(total_ms, 1),
        "tokens": token_count,
        "cached": False,
    })


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Production RAG</title>
<style>
:root{
  --bg:#08090d;
  --bg-grad-a:#141824;
  --bg-grad-b:#08090d;
  --surface:#0f121a;
  --surface-raised:#141824;
  --surface-sunken:#0b0e15;
  --border:#1e232e;
  --border-strong:#2a3140;
  --text:#e8ebf2;
  --text-soft:#b9c0d0;
  --muted:#6d7689;
  --muted-2:#4e5666;
  --accent:#5b8def;
  --accent-soft:rgba(91,141,239,.10);
  --accent-border:rgba(91,141,239,.30);
  --good:#3ecf8e;
  --good-soft:rgba(62,207,142,.10);
  --warn:#e0a94a;
  --warn-soft:rgba(224,169,74,.10);
  --bad:#f87171;
  --bad-soft:rgba(239,68,68,.10);
  --radius-lg:12px;
  --radius:9px;
  --radius-sm:6px;
  --shadow:0 1px 0 rgba(255,255,255,.02) inset, 0 8px 24px -12px rgba(0,0,0,.55);
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,sans-serif;
  background:radial-gradient(1200px 600px at 20% -10%,var(--bg-grad-a) 0%,var(--bg-grad-b) 55%,var(--bg) 100%);
  background-attachment:fixed;
  color:var(--text);
  font-size:14px;
  line-height:1.55;
  min-height:100vh;
  -webkit-font-smoothing:antialiased;
}
::selection{background:var(--accent-soft);color:var(--text)}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border-strong);border-radius:8px;border:2px solid transparent;background-clip:padding-box}
::-webkit-scrollbar-thumb:hover{background:var(--muted-2);background-clip:padding-box;border:2px solid transparent}

.app{
  max-width:1280px;
  margin:0 auto;
  padding:1.75rem 1.5rem 3rem;
}

.topbar{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:1rem;
  padding-bottom:1.25rem;
  border-bottom:1px solid var(--border);
  margin-bottom:1.5rem;
}
.brand{display:flex;align-items:center;gap:.85rem}
.brand-mark{
  width:34px;height:34px;border-radius:9px;
  background:linear-gradient(135deg,var(--accent) 0%,#7aa2ff 100%);
  box-shadow:0 6px 18px -6px rgba(91,141,239,.5), 0 1px 0 rgba(255,255,255,.15) inset;
  position:relative;
}
.brand-mark::after{
  content:"";position:absolute;inset:9px;border-radius:4px;
  background:linear-gradient(135deg,rgba(255,255,255,.9),rgba(255,255,255,.55));
  mask:linear-gradient(135deg,transparent 45%,#000 45%);
  -webkit-mask:linear-gradient(135deg,transparent 45%,#000 45%);
}
.brand h1{margin:0;font-size:1.15rem;font-weight:600;letter-spacing:-.015em;line-height:1.2}
.brand p{margin:.15rem 0 0;font-size:.78rem;color:var(--muted);letter-spacing:.01em}

.status-pill{
  display:inline-flex;align-items:center;gap:.45rem;
  padding:.35rem .75rem;border-radius:999px;
  background:var(--good-soft);border:1px solid rgba(62,207,142,.28);
  color:var(--good);font-size:.72rem;font-weight:500;
  letter-spacing:.02em;white-space:nowrap;
}
.status-pill.offline{background:var(--bad-soft);border-color:rgba(239,68,68,.3);color:var(--bad)}
.status-dot{width:6px;height:6px;border-radius:50%;background:currentColor;box-shadow:0 0 8px currentColor;animation:pulse 2.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}

.layout{
  display:grid;
  grid-template-columns:minmax(0,1fr) 340px;
  gap:1.25rem;
  align-items:start;
}
@media(max-width:1000px){
  .layout{grid-template-columns:1fr}
}

.panel{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:var(--radius-lg);
  box-shadow:var(--shadow);
  margin-bottom:1rem;
  overflow:hidden;
}
.panel-head{
  display:flex;align-items:center;justify-content:space-between;gap:.75rem;
  padding:.75rem 1rem;
  border-bottom:1px solid var(--border);
  background:linear-gradient(180deg,rgba(255,255,255,.012),rgba(255,255,255,0));
}
.panel-title{
  font-size:.68rem;font-weight:600;letter-spacing:.14em;
  text-transform:uppercase;color:var(--text-soft);
}
.panel-hint{
  font-size:.7rem;color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:60%;
}
.panel-body{padding:1rem}

.doc-strip{
  display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;
  padding:1rem;
}
.file-input{
  position:relative;display:inline-flex;align-items:center;gap:.55rem;
  padding:.55rem .9rem;background:var(--surface-sunken);
  border:1px dashed var(--border-strong);border-radius:var(--radius);
  cursor:pointer;transition:all .18s ease;flex:1;min-width:200px;
  font-size:.85rem;color:var(--text-soft);min-height:38px;
}
.file-input:hover{border-color:var(--accent);background:var(--surface-raised);color:var(--text)}
.file-input input[type="file"]{display:none}
.file-input .file-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.file-input.has-file{border-style:solid;border-color:var(--accent-border);background:var(--accent-soft);color:#7aa2ff}

.doc-chips{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.75rem;padding:0 1rem 1rem}
.doc-chip{
  display:inline-flex;align-items:center;gap:.4rem;
  padding:.3rem .6rem;border-radius:var(--radius-sm);
  background:var(--surface-sunken);border:1px solid var(--border);
  font-size:.72rem;color:var(--text-soft);
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
}
.doc-chip .doc-name{color:var(--text);max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.doc-chip .doc-count{color:var(--muted)}

.query-hero{padding:1rem}
.query-row{display:flex;gap:.5rem;align-items:stretch}
.query-input{
  flex:1;min-width:0;
  background:var(--surface-sunken);
  border:1px solid var(--border-strong);
  color:var(--text);
  padding:.7rem .95rem;
  border-radius:var(--radius);
  font-family:inherit;font-size:.92rem;
  transition:all .15s ease;
}
.query-input::placeholder{color:var(--muted-2)}
.query-input:hover{border-color:var(--muted-2)}
.query-input:focus{outline:none;border-color:var(--accent);background:var(--surface-raised);box-shadow:0 0 0 3px var(--accent-soft)}

.btn{
  display:inline-flex;align-items:center;justify-content:center;
  padding:.6rem 1.1rem;border-radius:var(--radius);
  border:1px solid transparent;font-weight:500;font-size:.85rem;
  cursor:pointer;font-family:inherit;transition:all .15s ease;
  min-height:38px;white-space:nowrap;
}
.btn-primary{
  background:linear-gradient(180deg,#5b8def 0%,#4a7bdc 100%);
  border-color:#4a7bdc;color:#fff;
  box-shadow:0 4px 12px -4px rgba(91,141,239,.4), 0 1px 0 rgba(255,255,255,.1) inset;
}
.btn-primary:hover{background:linear-gradient(180deg,#6a99f2 0%,#5488e6 100%)}
.btn-primary:disabled{opacity:.45;cursor:not-allowed;box-shadow:none}

.meta{
  display:flex;gap:.4rem;flex-wrap:wrap;
  margin:.9rem 0 .6rem;min-height:1.85rem;
  align-items:center;
}
.badge{
  display:inline-flex;align-items:center;gap:.35rem;
  font-size:.65rem;font-weight:600;letter-spacing:.1em;
  text-transform:uppercase;padding:.28rem .6rem;border-radius:var(--radius-sm);
  border:1px solid transparent;
}
.badge::before{content:"";width:5px;height:5px;border-radius:50%;background:currentColor}
.badge.cache{color:#7aa2ff;background:var(--accent-soft);border-color:var(--accent-border)}
.badge.fresh{color:var(--good);background:var(--good-soft);border-color:rgba(62,207,142,.28)}
.badge.err{color:var(--bad);background:var(--bad-soft);border-color:rgba(239,68,68,.3)}
.badge.warn{color:var(--warn);background:var(--warn-soft);border-color:rgba(224,169,74,.3)}
.metric{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.68rem;color:var(--muted);
  background:var(--surface-sunken);
  border:1px solid var(--border);
  padding:.28rem .55rem;border-radius:var(--radius-sm);
  white-space:nowrap;max-width:340px;overflow:hidden;text-overflow:ellipsis;
}
.metric strong{color:var(--text);font-weight:600}

.output{
  background:var(--surface-sunken);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:1rem 1.1rem;
  min-height:260px;max-height:600px;overflow-y:auto;
  font-size:.92rem;line-height:1.7;
  white-space:pre-wrap;word-break:break-word;
  color:var(--text);position:relative;
}
.output:empty::before{
  content:"Answers will appear here.";
  color:var(--muted-2);font-style:italic;
}
.output.streaming::after{
  content:"▌";color:var(--accent);
  margin-left:2px;animation:blink 1s steps(2) infinite;
  font-weight:400;
}
@keyframes blink{0%,49%{opacity:1}50%,100%{opacity:0}}
.output.flash{animation:flash .55s ease}
@keyframes flash{
  0%{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
  100%{border-color:var(--border);box-shadow:none}
}

.sidebar{display:flex;flex-direction:column;gap:1rem;position:sticky;top:1.5rem}
@media(max-width:1000px){.sidebar{position:static}}

.metric-grid{
  display:grid;grid-template-columns:1fr 1fr;gap:.5rem;
  padding:0;
}
.metric-cell{
  padding:.7rem .8rem;
  background:var(--surface-sunken);
  border:1px solid var(--border);
  border-radius:var(--radius);
}
.metric-cell .label{
  font-size:.64rem;text-transform:uppercase;letter-spacing:.12em;
  color:var(--muted);font-weight:600;margin-bottom:.25rem;
}
.metric-cell .value{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.95rem;color:var(--text);font-weight:600;
  line-height:1.2;
}
.metric-cell .value.good{color:var(--good)}
.metric-cell .value.warn{color:var(--warn)}
.metric-cell .value.accent{color:#7aa2ff}
.metric-cell .value .sub{
  font-size:.7rem;color:var(--muted);font-weight:400;
}

.threshold-block{padding:1rem}
.threshold-top{
  display:flex;justify-content:space-between;align-items:center;
  margin-bottom:.55rem;
}
.threshold-label{
  font-size:.72rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.1em;font-weight:600;
}
.threshold-value{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.85rem;color:#7aa2ff;font-weight:600;
  padding:.15rem .5rem;border-radius:var(--radius-sm);
  background:var(--accent-soft);border:1px solid var(--accent-border);
}
.slider{
  -webkit-appearance:none;appearance:none;
  width:100%;height:6px;border-radius:3px;
  background:linear-gradient(90deg,var(--border-strong) 0%,var(--border) 100%);
  outline:none;cursor:pointer;
}
.slider::-webkit-slider-thumb{
  -webkit-appearance:none;appearance:none;
  width:18px;height:18px;border-radius:50%;
  background:linear-gradient(180deg,#7aa2ff,#5b8def);
  border:2px solid #0b0d12;
  box-shadow:0 0 0 1px #7aa2ff, 0 2px 8px rgba(91,141,239,.4);
  cursor:pointer;transition:transform .12s ease;
}
.slider::-webkit-slider-thumb:hover{transform:scale(1.1)}
.slider::-moz-range-thumb{
  width:16px;height:16px;border-radius:50%;
  background:linear-gradient(180deg,#7aa2ff,#5b8def);
  border:2px solid #0b0d12;
  box-shadow:0 0 0 1px #7aa2ff, 0 2px 8px rgba(91,141,239,.4);
  cursor:pointer;
}
.slider-labels{
  display:flex;justify-content:space-between;
  font-size:.65rem;color:var(--muted-2);
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  margin-top:.4rem;text-transform:uppercase;letter-spacing:.08em;
}

.near-misses{padding:1rem;display:flex;flex-direction:column;gap:.65rem}
.near-miss-item{
  padding:.65rem .75rem;
  background:var(--surface-sunken);
  border:1px solid var(--border);
  border-radius:var(--radius);
}
.near-miss-head{
  display:flex;justify-content:space-between;align-items:center;
  gap:.5rem;margin-bottom:.35rem;
}
.near-miss-score{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.75rem;color:var(--warn);font-weight:600;
}
.near-miss-query{
  font-size:.72rem;color:var(--text-soft);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  flex:1;text-align:right;
}
.near-miss-bar{
  position:relative;height:4px;
  background:var(--border);border-radius:2px;overflow:visible;
}
.near-miss-fill{
  position:absolute;left:0;top:0;bottom:0;
  background:linear-gradient(90deg,var(--warn-soft),var(--warn));
  border-radius:2px;
}
.near-miss-threshold{
  position:absolute;top:-3px;bottom:-3px;width:2px;
  background:var(--accent);border-radius:1px;
  box-shadow:0 0 6px rgba(91,141,239,.6);
}
.empty-hint{
  padding:1.5rem 1rem;text-align:center;
  color:var(--muted-2);font-size:.78rem;font-style:italic;
}

.no-docs{
  padding:0 1rem 1rem;font-size:.75rem;color:var(--muted-2);font-style:italic;
}

.upload-status{
  padding:0 1rem 1rem;font-size:.75rem;color:var(--muted);
  min-height:1rem;
}
.upload-status.ok{color:var(--good)}
.upload-status.err{color:var(--bad)}
</style>
</head>
<body>
<div class="app">

  <header class="topbar">
    <div class="brand">
      <div class="brand-mark"></div>
      <div>
        <h1>Production RAG</h1>
        <p>FastAPI &middot; SSE streaming &middot; semantic cache</p>
      </div>
    </div>
    <span class="status-pill" id="status-pill">
      <span class="status-dot"></span>
      <span id="status-text">Checking</span>
    </span>
  </header>

  <div class="layout">

    <main>

      <section class="panel">
        <div class="panel-head">
          <span class="panel-title">Documents</span>
          <span class="panel-hint" id="doc-summary">No documents indexed</span>
        </div>
        <div class="doc-strip">
          <label class="file-input" id="file-wrap">
            <input type="file" id="file" accept=".pdf">
            <span class="file-label" id="file-label">Choose a PDF to index</span>
          </label>
          <button onclick="upload()" id="upload-btn" class="btn btn-primary">Index</button>
        </div>
        <div class="upload-status" id="upload-status"></div>
        <div class="doc-chips" id="doc-chips"></div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <span class="panel-title">Query</span>
          <span class="panel-hint">Press Enter to send</span>
        </div>
        <div class="query-hero">
          <div class="query-row">
            <input type="text" id="query" class="query-input" placeholder="Ask a question about your documents...">
            <button onclick="ask()" id="ask-btn" class="btn btn-primary">Send</button>
          </div>
          <div class="meta" id="meta"></div>
          <div class="output" id="output"></div>
        </div>
      </section>

    </main>

    <aside class="sidebar">

      <section class="panel">
        <div class="panel-head">
          <span class="panel-title">Semantic cache</span>
          <span class="panel-hint" id="cache-threshold-label">threshold 0.85</span>
        </div>
        <div class="panel-body">
          <div class="metric-grid" id="cache-stats"></div>
        </div>
        <div class="threshold-block">
          <div class="threshold-top">
            <span class="threshold-label">Similarity threshold</span>
            <span class="threshold-value" id="threshold-value">0.85</span>
          </div>
          <input type="range" id="threshold-slider" class="slider" min="0.60" max="0.99" step="0.01" value="0.85" oninput="updateThreshold(this.value)">
          <div class="slider-labels"><span>loose</span><span>strict</span></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <span class="panel-title">Near misses</span>
          <span class="panel-hint" id="near-miss-count">0</span>
        </div>
        <div id="near-misses" class="near-misses">
          <div class="empty-hint">Queries that land just below the threshold will appear here.</div>
        </div>
      </section>

    </aside>

  </div>
</div>

<script>
async function refreshHealth(){
  try{
    const [h, docs] = await Promise.all([
      fetch('/health').then(r=>r.json()),
      fetch('/documents').then(r=>r.json()).catch(()=>({documents:[], total_chunks:0}))
    ]);
    const pill = document.getElementById('status-pill');
    const text = document.getElementById('status-text');
    if(h.ollama_ok){
      pill.className = 'status-pill';
      text.textContent = 'Ollama connected';
    }else{
      pill.className = 'status-pill offline';
      text.textContent = 'Ollama offline';
    }
    const summary = document.getElementById('doc-summary');
    const chips = document.getElementById('doc-chips');
    if(!docs.documents || docs.documents.length === 0){
      summary.textContent = 'No documents indexed';
      chips.innerHTML = '';
    }else{
      summary.textContent = `${docs.documents.length} file${docs.documents.length===1?'':'s'} · ${docs.total_chunks} chunks`;
      chips.innerHTML = docs.documents.map(d =>
        `<span class="doc-chip"><span class="doc-name">${escapeHtml(d.source)}</span><span class="doc-count">${d.chunks}</span></span>`
      ).join('');
    }
  }catch(e){
    document.getElementById('status-text').textContent = 'Error';
  }
}

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function refreshCache(){
  try{
    const r = await fetch('/cache/stats');
    const d = await r.json();
    const stats = document.getElementById('cache-stats');
    const hitRate = (d.hit_rate * 100).toFixed(1);
    stats.innerHTML =
      `<div class="metric-cell"><div class="label">Hit rate</div><div class="value accent">${hitRate}<span class="sub"> %</span></div></div>` +
      `<div class="metric-cell"><div class="label">Entries</div><div class="value">${d.size}<span class="sub"> / ${d.max_size}</span></div></div>` +
      `<div class="metric-cell"><div class="label">Hits</div><div class="value good">${d.hits}</div></div>` +
      `<div class="metric-cell"><div class="label">Misses</div><div class="value">${d.misses}</div></div>` +
      `<div class="metric-cell"><div class="label">Near misses</div><div class="value warn">${d.near_misses}</div></div>` +
      `<div class="metric-cell"><div class="label">Saved</div><div class="value">${(d.total_saved_ms/1000).toFixed(1)}<span class="sub"> s</span></div></div>`;
    document.getElementById('threshold-value').textContent = d.threshold.toFixed(2);
    document.getElementById('cache-threshold-label').textContent = `threshold ${d.threshold.toFixed(2)}`;
    document.getElementById('threshold-slider').value = d.threshold;
  }catch(e){}
}

async function refreshNearMisses(){
  try{
    const r = await fetch('/cache/near-misses');
    const d = await r.json();
    const el = document.getElementById('near-misses');
    const count = document.getElementById('near-miss-count');
    count.textContent = String(d.entries.length);
    if(!d.entries.length){
      el.innerHTML = '<div class="empty-hint">Queries that land just below the threshold will appear here.</div>';
      return;
    }
    const threshold = d.threshold;
    el.innerHTML = d.entries.slice(0, 6).map(e => {
      const pct = Math.min(100, e.similarity * 100);
      const thrPct = Math.min(100, threshold * 100);
      return `<div class="near-miss-item">
        <div class="near-miss-head">
          <span class="near-miss-score">${(e.similarity*100).toFixed(1)}%</span>
          <span class="near-miss-query" title="${escapeHtml(e.cached_query)}">${escapeHtml(e.cached_query.slice(0,48))}${e.cached_query.length>48?'…':''}</span>
        </div>
        <div class="near-miss-bar">
          <div class="near-miss-fill" style="width:${pct}%"></div>
          <div class="near-miss-threshold" style="left:calc(${thrPct}% - 1px)"></div>
        </div>
      </div>`;
    }).join('');
  }catch(e){}
}

async function updateThreshold(v){
  const val = parseFloat(v);
  document.getElementById('threshold-value').textContent = val.toFixed(2);
  try{
    await fetch('/cache/threshold', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({threshold: val})
    });
  }catch(e){}
}

document.getElementById('file').addEventListener('change', function(){
  const wrap = document.getElementById('file-wrap');
  const label = document.getElementById('file-label');
  if(this.files && this.files[0]){
    label.textContent = this.files[0].name;
    wrap.classList.add('has-file');
  }else{
    label.textContent = 'Choose a PDF to index';
    wrap.classList.remove('has-file');
  }
});

async function upload(){
  const f = document.getElementById('file').files[0];
  if(!f) return;
  const btn = document.getElementById('upload-btn');
  const status = document.getElementById('upload-status');
  btn.disabled = true;
  status.className = 'upload-status';
  status.textContent = 'Uploading and embedding...';
  const form = new FormData();
  form.append('file', f);
  try{
    const r = await fetch('/upload', {method:'POST', body: form});
    const d = await r.json();
    if(r.ok){
      status.className = 'upload-status ok';
      status.textContent = `Indexed ${d.chunks} chunks from ${d.pages} pages. Total: ${d.total_chunks} chunks.`;
      refreshHealth();
    }else{
      status.className = 'upload-status err';
      status.textContent = `Error: ${d.detail || 'Upload failed'}`;
    }
  }catch(e){
    status.className = 'upload-status err';
    status.textContent = `Error: ${e.message}`;
  }
  btn.disabled = false;
}

function ask(){
  const q = document.getElementById('query').value.trim();
  if(!q) return;
  const btn = document.getElementById('ask-btn');
  const meta = document.getElementById('meta');
  const output = document.getElementById('output');
  btn.disabled = true;
  meta.innerHTML = '';
  output.textContent = '';
  output.className = 'output streaming';
  let firstToken = null;
  let full = '';
  let tokenChunks = 0;
  let completed = false;
  let isCache = false;

  const es = new EventSource(`/chat/stream?query=${encodeURIComponent(q)}`);

  es.addEventListener('meta', e => {
    const d = JSON.parse(e.data);
    if(d.type === 'cache_hit'){
      isCache = true;
      meta.innerHTML =
        `<span class="badge cache">Cache hit</span>` +
        `<span class="metric"><strong>${(d.similarity*100).toFixed(1)}%</strong> similarity</span>` +
        `<span class="metric">threshold <strong>${(d.threshold*100).toFixed(0)}%</strong></span>` +
        `<span class="metric">saved <strong>${(d.saved_ms/1000).toFixed(2)}s</strong></span>` +
        `<span class="metric" title="${escapeHtml(d.cached_query)}">match: ${escapeHtml(d.cached_query.slice(0,42))}${d.cached_query.length>42?'…':''}</span>`;
    }else if(d.type === 'fresh'){
      meta.innerHTML =
        `<span class="badge fresh">Fresh</span>` +
        `<span class="metric">context <strong>${d.context_chars}</strong> chars</span>` +
        `<span class="metric">threshold <strong>${(d.threshold*100).toFixed(0)}%</strong></span>`;
      if(d.nearest){
        const sim = d.nearest.similarity * 100;
        const thr = d.threshold * 100;
        const cls = d.nearest.similarity >= d.threshold * 0.95 ? 'warn' : '';
        meta.innerHTML +=
          `<span class="badge ${cls}">nearest ${sim.toFixed(1)}% &lt; ${thr.toFixed(0)}%</span>` +
          `<span class="metric" title="${escapeHtml(d.nearest.cached_query)}">match: ${escapeHtml(d.nearest.cached_query.slice(0,32))}${d.nearest.cached_query.length>32?'…':''}</span>`;
      }
    }else if(d.type === 'error'){
      meta.innerHTML = `<span class="badge err">Error</span><span class="metric">${escapeHtml(d.message)}</span>`;
      output.className = 'output';
      output.textContent = d.message;
    }
  });

  es.addEventListener('ttft', e => {
    const d = JSON.parse(e.data);
    if(firstToken === null){
      firstToken = d.ttft_ms;
      meta.innerHTML += `<span class="metric">ttft <strong>${d.ttft_ms.toFixed(0)}ms</strong></span>`;
    }
  });

  es.addEventListener('token', e => {
    const d = JSON.parse(e.data);
    if(firstToken === null){
      firstToken = 1;
      meta.innerHTML += `<span class="metric">streaming</span>`;
    }
    full += d.text;
    tokenChunks += 1;
    output.textContent = full;
    if(tokenChunks === 1 && isCache){
      output.classList.add('flash');
      setTimeout(()=>output.classList.remove('flash'), 600);
    }
  });

  es.addEventListener('done', e => {
    completed = true;
    const d = JSON.parse(e.data);
    output.className = 'output';
    meta.innerHTML += `<span class="metric">total <strong>${d.total_ms.toFixed(0)}ms</strong></span>`;
    es.close();
    btn.disabled = false;
    refreshCache();
    refreshNearMisses();
  });

  es.addEventListener('stream_error', e => {
    completed = true;
    const d = JSON.parse(e.data);
    meta.innerHTML += `<span class="badge err">${escapeHtml(d.message)}</span>`;
    output.className = 'output';
    es.close();
    btn.disabled = false;
  });

  es.onerror = () => {
    if(!completed) meta.innerHTML += `<span class="badge err">Connection lost</span>`;
    output.className = 'output';
    es.close();
    btn.disabled = false;
  };
}

document.getElementById('query').addEventListener('keydown', e => { if(e.key === 'Enter') ask(); });

refreshHealth();
refreshCache();
refreshNearMisses();
setInterval(refreshCache, 5000);
setInterval(refreshNearMisses, 5000);
setInterval(refreshHealth, 15000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")