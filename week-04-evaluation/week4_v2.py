from __future__ import annotations

import argparse
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
SLEEP_BETWEEN_TESTS = 0.2
TOP_K = 3

HIT_RATE_THRESHOLD = 0.90
FAITHFULNESS_THRESHOLD = 0.95
RELEVANCE_THRESHOLD = 0.90
JUDGE_AGREEMENT_THRESHOLD = 0.85

OUTPUT_DIR = Path("eval_results")

PREFERRED_GENERATOR = ["llama3.2", "llama3.1", "llama3", "mistral", "qwen2.5", "phi3"]
PREFERRED_JUDGE = ["mistral", "phi3", "qwen2.5", "llama3.1", "llama3.2"]


# KNOWLEDGE BASE

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


# GOLDEN DATASET

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


# JUDGE CALIBRATION SET

@dataclass
class CalibrationCase:
    context: str
    answer: str
    human_verdict: int
    note: str


CALIBRATION_SET: List[CalibrationCase] = [
    CalibrationCase(
        context="Employees receive a $100 monthly wellness stipend.",
        answer="The wellness stipend is $100 per month.",
        human_verdict=1,
        note="Direct paraphrase of the same amount and cadence.",
    ),
    CalibrationCase(
        context="The 401(k) match is 4% of base salary, capped at $12,000 per year.",
        answer="Acme matches 4% of your salary up to a maximum of $12,000 annually.",
        human_verdict=1,
        note="All three facts preserved: 4%, base salary, $12,000 cap.",
    ),
    CalibrationCase(
        context="Employees accrue 20 days PTO, 11 company holidays, and 5 days of bereavement leave.",
        answer="You accrue 20 PTO days per year.",
        human_verdict=1,
        note="Partial answer, but the stated fact is fully supported.",
    ),
    CalibrationCase(
        context="A deployment freeze applies during the last two weeks of December.",
        answer="There is a code freeze in late December.",
        human_verdict=1,
        note="Synonym swap preserves the meaning.",
    ),
    CalibrationCase(
        context="Cafeteria hours are 11am to 2pm.",
        answer="I don't know based on the provided context.",
        human_verdict=1,
        note="Declining to answer is faithful to unrelated context.",
    ),
    CalibrationCase(
        context=("Remote workers get a $100 wellness stipend. The stipend is "
                 "disbursed via the Benefits Portal on the first payroll cycle "
                 "of each month."),
        answer=("Remote workers receive $100 per month for wellness, paid "
                "through the Benefits Portal."),
        human_verdict=1,
        note="Combines two facts from context without adding anything.",
    ),
    CalibrationCase(
        context="Business class flights require 8 or more hours of scheduled air time.",
        answer="Business class flights are for long-haul flights.",
        human_verdict=1,
        note="Less precise but not incorrect.",
    ),
    CalibrationCase(
        context="Unused PTO may be carried over up to 5 days.",
        answer="You can carry over 5 days of unused PTO.",
        human_verdict=1,
        note="'Up to' is dropped but the ceiling is preserved.",
    ),
    CalibrationCase(
        context=("Document A says the 401(k) match is 4% of salary. "
                 "Document B says PTO accrual is 20 days per year."),
        answer="The 401(k) match is 4% of salary.",
        human_verdict=1,
        note="Uses only one source; still supported by the context block.",
    ),
    CalibrationCase(
        context="",
        answer="I don't know based on the provided context.",
        human_verdict=1,
        note="Empty context, honest decline.",
    ),
    CalibrationCase(
        context="The wellness stipend is $100 per month.",
        answer="The wellness allowance is $100 monthly.",
        human_verdict=1,
        note="Synonym swap: stipend/allowance, per month/monthly.",
    ),
    CalibrationCase(
        context="Employees may work remotely up to 3 days per week.",
        answer="Employees can work remotely 3 days a week or fewer.",
        human_verdict=1,
        note="Reworded but retains the ceiling semantics.",
    ),

    # ------------------ Unfaithful (verdict 0) ------------------
    CalibrationCase(
        context="The wellness stipend is $100 per month.",
        answer="The wellness stipend is $150 per month.",
        human_verdict=0,
        note="Number inflated.",
    ),
    CalibrationCase(
        context="Employees accrue 20 days of PTO per year.",
        answer="You get 15 PTO days annually.",
        human_verdict=0,
        note="Number deflated.",
    ),
    CalibrationCase(
        context="The 401(k) match is 4% of base salary.",
        answer="Acme offers an employee stock purchase plan.",
        human_verdict=0,
        note="Wrong entity, topic shift.",
    ),
    CalibrationCase(
        context="Business class is allowed for flights over 8 hours.",
        answer=("Business class is allowed for flights over 8 hours and "
                "includes lounge access."),
        human_verdict=0,
        note="Adds a lounge-access detail not in context.",
    ),
    CalibrationCase(
        context="Employees are eligible for the 401(k) match after 90 days.",
        answer="You're eligible for the match after 30 days.",
        human_verdict=0,
        note="Wrong eligibility period.",
    ),
    CalibrationCase(
        context="The 401(k) match is 4% of salary. PTO accrual is 20 days per year.",
        answer="You accrue 4% PTO and get a 20-day 401(k) match.",
        human_verdict=0,
        note="Facts crossed between documents.",
    ),
    CalibrationCase(
        context="Unused PTO may be carried over up to 5 days.",
        answer="Per Section 4.2 of the handbook, carryover is capped at 5 days.",
        human_verdict=0,
        note="Section reference is fabricated.",
    ),
    CalibrationCase(
        context="The deployment freeze runs through the last two weeks of December.",
        answer=("Deploys are frozen in late December, which aligns with "
                "industry practice."),
        human_verdict=0,
        note="Adds outside commentary not in context.",
    ),
    CalibrationCase(
        context="Unused PTO may be carried over up to 5 days.",
        answer="All unused PTO carries over.",
        human_verdict=0,
        note="Reverses the cap.",
    ),
    CalibrationCase(
        context="Business class requires flights over 8 hours.",
        answer="Business class requires flights of at least 8 hours.",
        human_verdict=0,
        note="'Over 8' and 'at least 8' differ at the boundary.",
    ),
    CalibrationCase(
        context="The wellness stipend is $100 per month.",
        answer="The wellness stipend is $100 per month, paid on the 15th.",
        human_verdict=0,
        note="Disbursement date invented.",
    ),
    CalibrationCase(
        context=("Employees are eligible for the 401(k) match after 90 days "
                 "of continuous service."),
        answer="Eligibility begins after 60 days.",
        human_verdict=0,
        note="Wrong eligibility window.",
    ),
    CalibrationCase(
        context="Business class is allowed for flights over 8 hours.",
        answer=("Business class is allowed for flights over 8 hours, and first "
                "class for flights over 12 hours."),
        human_verdict=0,
        note="First-class policy does not exist in context.",
    ),
    CalibrationCase(
        context="The 401(k) match is 4% of salary. PTO is 20 days.",
        answer="You get a 20% match and 4 days PTO.",
        human_verdict=0,
        note="Swapped numbers between facts.",
    ),
    CalibrationCase(
        context="The deployment freeze runs through the last two weeks of December.",
        answer="There is a hiring freeze in late December.",
        human_verdict=0,
        note="Wrong policy domain.",
    ),
    CalibrationCase(
        context=("If approved by a director, business class is permitted on "
                 "flights over 8 hours."),
        answer="If the flight is over 8 hours, business class is automatically approved.",
        human_verdict=0,
        note="Reverses the conditional.",
    ),
    CalibrationCase(
        context="The 401(k) match is capped at $12,000 per year.",
        answer="There is no cap on the 401(k) match.",
        human_verdict=0,
        note="Contradicts the cap.",
    ),
    CalibrationCase(
        context="The wellness stipend is $100 per month.",
        answer=("The wellness stipend is $100 per month, and the company also "
                "offers a gym subsidy."),
        human_verdict=0,
        note="Second claim is not in context.",
    ),
]


# MODEL RESOLUTION

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
            if name == candidate or name.split(":")[0] == candidate:
                return name
    return available[0] if available else None


def resolve_models(generator_override: Optional[str], judge_override: Optional[str]) -> tuple[str, str]:
    available = list_available_models()

    if not available:
        print("No Ollama models detected. Start the server with `ollama serve` "
              "and pull a model, e.g. `ollama pull llama3.2`.")
        sys.exit(1)

    generator = generator_override or pick_model(available, PREFERRED_GENERATOR)
    judge = judge_override or pick_model(available, PREFERRED_JUDGE)

    if generator not in available and generator_override:
        print(f"Warning: requested generator `{generator}` is not installed. "
              f"Available: {', '.join(available)}")
        sys.exit(1)
    if judge not in available and judge_override:
        print(f"Warning: requested judge `{judge}` is not installed. "
              f"Available: {', '.join(available)}")
        sys.exit(1)

    return generator, judge


# MOCK RAG PIPELINE

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


# JUDGE UTILITIES

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


# JUDGES

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


# JUDGE CALIBRATION

def run_calibration(judge_model: str) -> Dict[str, Any]:
    print("=" * 64)
    print("JUDGE CALIBRATION")
    print("=" * 64)
    print(f"Judge model   : {judge_model}")
    print(f"Labeled cases : {len(CALIBRATION_SET)}")
    print("=" * 64)
    print()

    agreements = 0
    scored = 0
    unparsed = 0
    disagreements: List[Dict[str, Any]] = []

    for i, case in enumerate(CALIBRATION_SET, start=1):
        judge_verdict = judge_faithfulness(case.context, case.answer, judge_model)

        if judge_verdict is None:
            unparsed += 1
            print(f"[{i:>2}] unparsed   human={case.human_verdict}  judge=<none>")
            continue

        scored += 1
        match = judge_verdict == case.human_verdict

        if match:
            agreements += 1
            print(f"[{i:>2}] agree      human={case.human_verdict}  judge={judge_verdict}")
        else:
            print(f"[{i:>2}] DISAGREE   human={case.human_verdict}  judge={judge_verdict}")
            disagreements.append({
                "index": i,
                "context": case.context,
                "answer": case.answer,
                "human_verdict": case.human_verdict,
                "judge_verdict": judge_verdict,
                "note": case.note,
            })

        time.sleep(SLEEP_BETWEEN_TESTS)

    agreement_rate = agreements / scored if scored else 0.0
    passed = agreement_rate >= JUDGE_AGREEMENT_THRESHOLD

    print()
    print("-" * 64)
    print(f"Agreement : {agreements}/{scored}  ({agreement_rate * 100:.1f}%)")
    if unparsed:
        print(f"Unparsed  : {unparsed} (excluded from agreement)")
    print(f"Threshold : {JUDGE_AGREEMENT_THRESHOLD * 100:.0f}%")

    if passed:
        print("Status    : PASS - judge verdicts are trustworthy for this suite.")
    else:
        print("Status    : FAIL - judge verdicts are unreliable.")
        print()
        print("Suggestion: swap the judge to a different model family rather than")
        print("            extending the prompt. Try one of: mistral, phi3, qwen2.5.")
        print(f"            Current judge: {judge_model}")

    print("=" * 64)
    print()

    if disagreements:
        print("Disagreements")
        print("-" * 64)
        for d in disagreements:
            print(f"  case {d['index']}")
            print(f"    context        : {d['context'][:80]}")
            print(f"    answer         : {d['answer'][:80]}")
            print(f"    human verdict  : {d['human_verdict']}")
            print(f"    judge verdict  : {d['judge_verdict']}")
            print(f"    human note     : {d['note']}")
            print()

    return {
        "judge_model": judge_model,
        "agreement_rate": round(agreement_rate, 4),
        "agreements": agreements,
        "scored": scored,
        "unparsed": unparsed,
        "threshold": JUDGE_AGREEMENT_THRESHOLD,
        "passed": passed,
        "disagreements": disagreements,
    }


# RAG TRIAD SUITE

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


def summarize(results: List[TestResult], generator_model: str, judge_model: str,
              calibration: Dict[str, Any]) -> Dict[str, Any]:
    ok = [r for r in results if r.error is None]
    errored = [r for r in results if r.error is not None]

    hit_rate = safe_rate([r.hit for r in ok])
    faithfulness = safe_rate([r.faithfulness for r in ok])
    relevance = safe_rate([r.relevance for r in ok])

    gen_latencies = [r.generation_latency for r in ok if r.generation_latency]
    judge_latencies = [r.judge_latency for r in ok if r.judge_latency]

    triad_passed = (
        hit_rate >= HIT_RATE_THRESHOLD
        and faithfulness >= FAITHFULNESS_THRESHOLD
        and relevance >= RELEVANCE_THRESHOLD
    )

    return {
        "calibration": calibration,
        "config": {
            "generator_model": generator_model,
            "judge_model": judge_model,
            "top_k": TOP_K,
            "thresholds": {
                "hit_rate": HIT_RATE_THRESHOLD,
                "faithfulness": FAITHFULNESS_THRESHOLD,
                "relevance": RELEVANCE_THRESHOLD,
                "judge_agreement": JUDGE_AGREEMENT_THRESHOLD,
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
        "passed": triad_passed and calibration["passed"],
        "triad_passed": triad_passed,
        "results": [asdict(r) for r in results],
    }


def run_evaluation_suite(generator_model: str, judge_model: str) -> List[TestResult]:
    print("=" * 64)
    print("RAG TRIAD SUITE")
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

    return results


# REPORTING

def print_report(summary: Dict[str, Any]) -> None:
    m = summary["metrics"]
    t = summary["totals"]
    lat = summary["latency"]
    cfg = summary["config"]
    cal = summary["calibration"]

    print()
    print("=" * 64)
    print("REGRESSION REPORT")
    print("=" * 64)
    print(f"Generator       : {cfg['generator_model']}")
    print(f"Judge           : {cfg['judge_model']}")
    print(f"Test cases      : {t['test_cases']}  (ok {t['successful']}, errored {t['errored']})")
    print()
    print("Judge calibration")
    print(f"  Agreement     : {cal['agreements']}/{cal['scored']}  ({cal['agreement_rate'] * 100:>5.1f}%)  "
          f"{'PASS' if cal['passed'] else 'FAIL'}")
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
    elif not cal["passed"]:
        print("Pipeline Blocked. Judge failed calibration - triad scores are unreliable.")
        print("Fix the judge before tuning anything against these numbers.")
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


# MAIN

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG evaluation suite with judge calibration")
    parser.add_argument("--generator", help="Override generator model name")
    parser.add_argument("--judge", help="Override judge model name")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip the calibration step and run only the triad")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Run only the calibration step and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    generator_model, judge_model = resolve_models(args.generator, args.judge)

    calibration: Dict[str, Any] = {
        "judge_model": judge_model,
        "agreement_rate": None,
        "agreements": 0,
        "scored": 0,
        "unparsed": 0,
        "threshold": JUDGE_AGREEMENT_THRESHOLD,
        "passed": True,
        "disagreements": [],
        "skipped": True,
    }

    if not args.skip_calibration:
        try:
            calibration = run_calibration(judge_model)
            calibration["skipped"] = False
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 130
        except Exception as exc:
            print(f"Calibration failed: {exc}")
            return 1

        if args.calibrate_only:
            path = save_results({"calibration": calibration, "calibrate_only": True})
            print(f"Results written to {path.resolve()}")
            return 0 if calibration["passed"] else 2

        if not calibration["passed"]:
            print("Calibration failed. Continuing to the triad for reference,")
            print("but treat the faithfulness numbers with caution.")
            print()

    try:
        results = run_evaluation_suite(generator_model, judge_model)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        return 1

    summary = summarize(results, generator_model, judge_model, calibration)
    print_report(summary)

    path = save_results(summary)
    print(f"Results written to {path.resolve()}")

    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())