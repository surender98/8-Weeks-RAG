# 8-Week RAG Challenge: From Zero to Production-Grade Agentic Systems 🚀

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Local%20LLM-Ollama-black.svg)](https://ollama.com/)

An intensive, 8-week engineering sprint designed to build, benchmark, and scale **Retrieval-Augmented Generation (RAG)** systems from first principles.

Instead of relying on black-box abstractions, this repository implements each layer of the modern RAG stack step-by-step—progressing from pure vector math and local models to hybrid retrieval, automated evaluation suites, agentic tool routing, and self-healing corrective architectures.

---

## 🧭 Repository Architecture & Directory Layout

Each week lives in its own directory, structured as an independent, reproducible project with its own source code, documentation, and dependencies:

```text
.
├── README.md                           # Master repo overview & roadmap
├── week-01-rag-fundamentals/           # Naive RAG, Chunking, In-memory Vector Search & Local UI
├── week-02-hybrid-search/              # Sparse (BM25) + Dense retrieval with Reciprocal Rank Fusion (RRF)
├── week-03-reranking-metadata/         # Two-stage retrieval, Cross-Encoders & Metadata Pre-filtering
├── week-04-evaluation/                 # Golden datasets, RAGAS metrics & LLM-as-a-Judge test suites
├── week-05-query-transformation/       # HyDE, Multi-Query expansion & Step-back prompting
├── week-06-agentic-rag/                # ReAct pattern, Dynamic retrieval gating & Function calling
├── week-07-production-rag/             # FastAPI backend, Server-Sent Events (SSE) & Semantic Caching
└── week-08-advanced-architectures/     # Corrective RAG (CRAG) & Graph-enhanced retrieval capstone
```

---

## 🗺️ The 8-Week Curriculum & Roadmap

| Week | Phase | Focus Concepts | Primary Technologies |
| --- | --- | --- | --- |
| **Week 1** | **RAG Fundamentals** | Chunking strategies, Vector embeddings, Cosine similarity, Grounded prompting, Document parsing | Python, NumPy, Ollama (`llama3.2`, `nomic-embed-text`), Streamlit |
| **Week 2** | **Hybrid Search** | Sparse vs. Dense, BM25 algorithm, Score normalization, Reciprocal Rank Fusion (RRF) | Python, Rank-BM25, NumPy |
| **Week 3** | **Reranking & Metadata** | Bi-encoders vs. Cross-encoders, Two-stage retrieval (over-retrieve & rerank), Metadata filtering | Sentence-Transformers / Cohere / BGE-Reranker |
| **Week 4** | **Evaluation & Evals** | Retrieval metrics (Hit Rate@k, MRR), Generation metrics (Faithfulness, Relevance), Golden datasets | RAGAS, LLM-as-a-Judge, Pytest |
| **Week 5** | **Query Transformation** | Multi-Query Expansion, HyDE (Hypothetical Document Embeddings), Step-Back Prompting, Sub-query Decomposition | LangChain / Custom orchestration, Local LLMs |
| **Week 6** | **Agentic RAG** | ReAct reasoning loops, Autonomous tool calling, Dynamic retrieval gating, Step budgeting | LangGraph / Native tool-calling loops |
| **Week 7** | **Production RAG** | Streaming endpoints (SSE), Exact & Semantic caching, Latency optimization, Observability | FastAPI, Redis / Qdrant, Uvicorn |
| **Week 8** | **Advanced Architectures** | Corrective RAG (CRAG), Self-RAG, Knowledge Graphs vs. Flat vector spaces, Fallback routing | NetworkX / Neo4j, CRAG workflow engines |

---

## 🚀 Getting Started

### 1. Global Prerequisites

* **Python 3.10+**
* **Git**
* **Ollama** (for local model execution): [Download Ollama](https://ollama.com/)

Pull the default local models used across the earlier modules:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/8-week-rag-challenge.git
cd 8-week-rag-challenge
```

### 3. Running an Individual Week

Navigate into any week's directory, follow its local `README.md` for specific package installations, and launch the entry point:

```bash
# Example: Running Week 1
cd week-01-rag-fundamentals
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Weekly Deep-Dive Prompts

Every week's folder includes a dedicated **LLM Study Prompt** designed to test your theoretical understanding before diving into the code:

* **Week 1:** Vector space geometry, chunking tradeoffs, and parametric vs. non-parametric memory.
* **Week 2:** BM25 term frequency saturation, length normalization, and rank fusion mechanics.
* **Week 3:** Cross-encoder attention mechanisms, compute bottlenecks, and stage candidate sizing.
* **Week 4:** Designing unbiased LLM judges, golden set curation, and statistical drift detection.
* **Week 5:** Vector alignment between questions and answers, and embedding distributions.
* **Week 6:** State machines for autonomous agents, memory boundaries, and infinite loop mitigations.
* **Week 7:** Time-to-First-Token (TTFT) reduction, semantic cache invalidation, and prompt injection hardening.
* **Week 8:** Graph traversal for multi-hop reasoning and deterministic fallback policies.

---

## 📌 Development Principles

1. **First-Principles First:** Understand the math and retrieval mechanics before abstracting them behind framework wrappers.
2. **Deterministic & Grounded:** Build systems with explicit guardrails, zero-temperature generation, and source attribution.
3. **Measure Everything:** If an architecture change doesn't yield measurable improvements on Hit Rate, MRR, or Faithfulness, it doesn't ship.
4. **Local & Private:** Prioritize architectures that can run locally on edge hardware before introducing cloud-hosted endpoints.

---
