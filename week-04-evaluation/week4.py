from __future__ import annotations

import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama


JUDGE_TEMPERATURE = 0.0
MAX_JUDGE_RETRIES = 3
RETRY_DELAY = 1.0
SLEEP_BETWEEN_TESTS = 0.3
TOP_K = 3

HIT_RATE_THRESHOLD = 0.90
FAITHFULNESS_THRESHOLD = 0.95
RELEVANCE_THRESHOLD = 0.90

OUTPUT_DIR = Path("eval_results")

PREFERRED_MODELS = [
    "llama3.2",
    "llama3.1",
    "llama3",
    "mistral",
    "qwen2.5",
    "phi3",
]


MOCK_DATABASE: Dict[str, str] = {
    "doc_001": "Acme Corp remote workers receive a $100 monthly wellness stipend. "
               "The stipend is disbursed through the Acme Benefits Portal on the "
               "first payroll cycle of each month.",
    "doc_002": "The company 401(k) match is 4% of the employee's base salary, "
               "capped at $12,000 per year. Employees are eligible after 90 days "
               "of continuous full-time service.",
    "doc_003": "Business class flights are only permitted for flights over 8 hours "
               "of scheduled air time. Prior approval from a director is required "
               "for any business class booking.",
    "doc_004": "Employees accrue 20 days of paid time off per year, plus 11 "
               "company holidays. Unused PTO may be carried over up to 5 days.",
    "doc_005": "A deployment freeze applies during the last two weeks of December "
               "and the final Friday of each month.",
}


@dataclass
class TestCase:
    query: str
    expected_doc_id: Optional[str]
    should_fail: bool = False
    tags: List[str] = field(default_factory=list)


GOLDEN_DATASET: List[TestCase] = [
    TestCase("How much is the wellness stipend?", "doc_001", tags=["factual", "single-hop"]),
    TestCase("What is the 401k match?", "doc_002", tags=["factual", "single-hop"]),
    TestCase("When can I fly business class?", "doc_003", tags=["factual", "single-hop"]),
    TestCase("How many PTO days do I get?", "doc_004", tags=["factual", "single-hop"]),
    TestCase("When is the deployment freeze?", "doc_005", tags=["factual", "single-hop"]),
    TestCase("What is the policy on stock options?", None, should_fail=True,
             tags=["hallucination-bait", "out-of-scope"]),
    TestCase("What is the CEO's home address?", None, should_fail=True,
             tags=["hallucination-bait", "out-of-scope"]),
]


def list_available_models() -> List[str]:
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


def pick_model(available: List[str], preferred: List[str]) -> Optional[str]:
    for candidate in preferred:
        for name in available:
            base = name.split(":")[0]
            if base == candidate or name == candidate:
                return name
    return available[0] if available else None


def resolve_models() -> tuple[str, str]:
    available = list_available_models()

    if not available:
        print("No Ollama models detected. Start the server with `ollama serve` "
              "and pull a model, e.g. `ollama pull llama3.2`.")
        sys.exit(1)

    chosen = pick_model(available, PREFERRED_MODELS)
    return chosen, chosen


def mock_retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, str]]:
    q = query.lower()

    keyword_map = [
        (["wellness", "stipend"], "doc_001"),
        (["401k", "401(k)", "retirement match"], "doc_002"),
        (["business class", "flight"], "doc_003"),
        (["pto", "paid time off", "vacation"], "doc_004"),
        (["deploy", "freeze"], "doc_005"),
    ]

    ranked_ids: List[str] = []
    for keywords, doc_id in keyword_map:
        if any(k in q for k in keywords):
            ranked_ids.append(doc_id)

    if not ranked_ids:
        ranked_ids = ["doc_001"]

    while len(ranked_ids) < top_k:
        for filler in ("doc_001", "doc_002", "doc_003"):
            if filler not in ranked_ids:
                ranked_ids.append(filler)
                break
        else:
            break

    return [{"id": d, "text": MOCK_DATABASE[d]} for d in ranked_ids[:top_k]]


def generate_answer(query: str, context: str, model: str) -> str:
    system = (
        "You are a factual assistant. Answer the user's question using ONLY the "
        "context provided. If the answer is not contained in the context, respond "
        "exactly: 'I don't know based on the provided context.' "
        "Do not add outside knowledge."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}"

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.0},
    )
    return response["message"]["content"].strip()


BINARY_PATTERN = re.compile(r"\b([01])\b")


def parse_binary_score(raw: str) -> Optional[int]:
    cleaned = raw.strip().lower()

    if cleaned in ("0", "1"):
        return int(cleaned)

    match = BINARY_PATTERN.search(cleaned)
    if match:
        return int(match.group(1))

    if "unfaithful" in cleaned or "irrelevant" in cleaned or "incorrect" in cleaned:
        return 0
    if "faithful" in cleaned or "relevant" in cleaned or "correct" in cleaned:
        return 1

    return None


def call_judge(prompt: str, judge_model: str) -> Optional[int]:
    for attempt in range(MAX_JUDGE_RETRIES):
        try:
            response = ollama.chat(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": JUDGE_TEMPERATURE},
            )
            score = parse_binary_score(response["message"]["content"])
            if score is not None:
                return score
        except Exception as exc:
            print(f"    judge attempt {attempt + 1} failed: {exc}")

        if attempt < MAX_JUDGE_RETRIES - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))

    return None


FAITHFULNESS_PROMPT = """You are a strict, impartial judge evaluating a RAG system.

CONTEXT PROVIDED TO THE SYSTEM:
{context}

GENERATED ANSWER:
{answer}

TASK:
Determine whether every factual claim in the GENERATED ANSWER is directly
supported by the CONTEXT. Any claim not present in the context, even if it
sounds plausible or true in the real world, counts as unfaithful.

If the answer declines to answer (e.g. "I don't know"), that is faithful.

Reply with exactly one character: 1 if faithful, 0 if unfaithful.
Do not include any explanation, punctuation, or other text."""


RELEVANCE_PROMPT = """You are a strict, impartial judge evaluating a RAG system.

USER QUESTION:
{query}

GENERATED ANSWER:
{answer}

TASK:
Determine whether the GENERATED ANSWER directly addresses the USER QUESTION.
An answer that declines to answer is NOT relevant (score 0).
An answer that is on-topic but factually wrong is still considered relevant
if it attempts to answer the question directly.

Reply with exactly one character: 1 if relevant, 0 if not relevant.
Do not include any explanation, punctuation, or other text."""


def judge_faithfulness(context: str, answer: str, judge_model: str) -> Optional[int]:
    lowered = answer.lower()
    if "i don't know" in lowered or "cannot answer" in lowered:
        return 1
    return call_judge(FAITHFULNESS_PROMPT.format(context=context, answer=answer), judge_model)


def judge_answer_relevance(query: str, answer: str, judge_model: str) -> Optional[int]:
    if "i don't know" in answer.lower():
        return 0
    return call_judge(RELEVANCE_PROMPT.format(query=query, answer=answer), judge_model)


def compute_hit(retrieved_ids: List[str], expected_id: Optional[str]) -> Optional[int]:
    if expected_id is None:
        return None
    return 1 if expected_id in retrieved_ids else 0


@dataclass
class TestResult:
    query: str
    tags: List[str]
    should_fail: bool
    retrieved_ids: List[str]
    answer: str
    hit: Optional[int] = None
    faithfulness: Optional[int] = None
    relevance: Optional[int] = None
    generation_latency: float = 0.0
    judge_latency: float = 0.0
    error: Optional[str] = None


def run_test_case(case: TestCase, generator_model: str, judge_model: str) -> TestResult:
    result = TestResult(
        query=case.query,
        tags=case.tags,
        should_fail=case.should_fail,
        retrieved_ids=[],
        answer="",
    )

    try:
        retrieved = mock_retrieve(case.query)
        result.retrieved_ids = [d["id"] for d in retrieved]
        context_text = "\n\n".join(d["text"] for d in retrieved)

        result.hit = compute_hit(result.retrieved_ids, case.expected_doc_id)

        gen_start = time.perf_counter()
        result.answer = generate_answer(case.query, context_text, generator_model)
        result.generation_latency = time.perf_counter() - gen_start

        judge_start = time.perf_counter()
        result.faithfulness = judge_faithfulness(context_text, result.answer, judge_model)
        result.relevance = judge_answer_relevance(case.query, result.answer, judge_model)
        result.judge_latency = time.perf_counter() - judge_start

    except Exception as exc:
        result.error = str(exc)

    return result


def safe_rate(values: List[Optional[int]]) -> float:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else 0.0


def summarize(results: List[TestResult], generator_model: str, judge_model: str) -> Dict[str, Any]:
    ok = [r for r in results if r.error is None]
    errored = [r for r in results if r.error is not None]

    hit_rate = safe_rate([r.hit for r in ok])
    faithfulness = safe_rate([r.faithfulness for r in ok])
    relevance = safe_rate([r.relevance for r in ok])

    gen_latencies = [r.generation_latency for r in ok if r.generation_latency]
    judge_latencies = [r.judge_latency for r in ok if r.judge_latency]

    passed = (
        hit_rate >= HIT_RATE_THRESHOLD
        and faithfulness >= FAITHFULNESS_THRESHOLD
        and relevance >= RELEVANCE_THRESHOLD
    )

    return {
        "config": {
            "generator_model": generator_model,
            "judge_model": judge_model,
            "top_k": TOP_K,
            "thresholds": {
                "hit_rate": HIT_RATE_THRESHOLD,
                "faithfulness": FAITHFULNESS_THRESHOLD,
                "relevance": RELEVANCE_THRESHOLD,
            },
        },
        "totals": {
            "test_cases": len(results),
            "successful": len(ok),
            "errored": len(errored),
        },
        "metrics": {
            "hit_rate": round(hit_rate, 4),
            "faithfulness": round(faithfulness, 4),
            "answer_relevance": round(relevance, 4),
        },
        "latency": {
            "generation_mean_s": round(statistics.mean(gen_latencies), 3) if gen_latencies else 0.0,
            "judge_mean_s": round(statistics.mean(judge_latencies), 3) if judge_latencies else 0.0,
        },
        "passed": passed,
        "results": [asdict(r) for r in results],
    }


def print_report(summary: Dict[str, Any]) -> None:
    m = summary["metrics"]
    t = summary["totals"]
    lat = summary["latency"]
    cfg = summary["config"]

    print()
    print("=" * 64)
    print("REGRESSION REPORT")
    print("=" * 64)
    print(f"Generator       : {cfg['generator_model']}")
    print(f"Judge           : {cfg['judge_model']}")
    print(f"Test cases      : {t['test_cases']}  (ok {t['successful']}, errored {t['errored']})")
    print()
    print("RAG Triad")
    print(f"  Hit Rate (Retrieval)      : {m['hit_rate'] * 100:>6.1f}%")
    print(f"  Faithfulness (Generation) : {m['faithfulness'] * 100:>6.1f}%")
    print(f"  Answer Relevance (Gen)    : {m['answer_relevance'] * 100:>6.1f}%")
    print()
    print("Latency")
    print(f"  Generation                : {lat['generation_mean_s']:.2f}s mean")
    print(f"  Judge                     : {lat['judge_mean_s']:.2f}s mean")
    print()

    if summary["passed"]:
        print("Pipeline Passed. Ready to deploy.")
    else:
        print("Pipeline Degraded. Do not ship.")
    print("=" * 64)


def save_results(summary: Dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"eval_{timestamp}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return path


def run_evaluation_suite(generator_model: str, judge_model: str) -> Dict[str, Any]:
    print("=" * 64)
    print("RAG EVALUATION SUITE")
    print("=" * 64)
    print(f"Generator : {generator_model}")
    print(f"Judge     : {judge_model}")
    print(f"Test cases: {len(GOLDEN_DATASET)}")
    print(f"Top-k     : {TOP_K}")
    print("=" * 64)
    print()

    results: List[TestResult] = []

    for i, case in enumerate(GOLDEN_DATASET, start=1):
        print(f"[{i}/{len(GOLDEN_DATASET)}] {case.query}")
        print(f"    tags      : {', '.join(case.tags)}")

        result = run_test_case(case, generator_model, judge_model)
        results.append(result)

        if result.error:
            print(f"    ERROR     : {result.error}")
        else:
            hit_label = "n/a" if result.hit is None else ("hit" if result.hit else "miss")
            faith_label = (
                "n/a" if result.faithfulness is None
                else ("faithful" if result.faithfulness else "unfaithful")
            )
            rel_label = (
                "n/a" if result.relevance is None
                else ("relevant" if result.relevance else "irrelevant")
            )

            print(f"    retrieved : {result.retrieved_ids}")
            print(f"    answer    : {result.answer[:110]}")
            print(f"    hit       : {hit_label}")
            print(f"    faithful  : {faith_label}")
            print(f"    relevance : {rel_label}")
            print(
                f"    latency   : gen {result.generation_latency:.2f}s | "
                f"judge {result.judge_latency:.2f}s"
            )

        print()
        time.sleep(SLEEP_BETWEEN_TESTS)

    return summarize(results, generator_model, judge_model)


def main() -> int:
    generator_model, judge_model = resolve_models()

    try:
        summary = run_evaluation_suite(generator_model, judge_model)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        return 1

    print_report(summary)

    path = save_results(summary)
    print(f"Results written to {path.resolve()}")

    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())