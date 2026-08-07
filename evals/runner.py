"""Evaluation runner for agentic RAG.

Usage:
    python -m evals.runner              # basic evaluation
    python -m evals.runner --deepeval   # with DeepEval metrics (requires API keys)

Exits with code 0 if all tests pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env.local for API keys (same as index.py does)
from dotenv import load_dotenv
load_dotenv(ROOT / ".env.local")

from api.agent import run_agent  # noqa: E402


# ---- Test Case Definition ---------------------------------------------------------------

class TestCase:
    def __init__(self, data: dict[str, Any]):
        self.query = data["query"]
        self.category = data.get("category", "uncategorized")

    def evaluate(self, result: dict[str, Any]) -> tuple[bool, list[str]]:
        """Return (passed, list_of_failure_reasons)."""
        failures = []

        # Basic sanity checks
        if not result.get("answer", "").strip():
            failures.append("Empty answer")

        if not result.get("steps"):
            failures.append("No steps executed")

        if not result.get("sources"):
            failures.append("No sources returned")

        if result.get("iterations", 0) == 0:
            failures.append("Zero iterations")

        return len(failures) == 0, failures


def load_test_cases(path: Path) -> list[TestCase]:
    cases = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(TestCase(json.loads(line)))
    return cases


# ---- DeepEval Integration (Optional) ----------------------------------------------------

def run_deepeval(query: str, answer: str, sources: list[dict]) -> dict[str, float] | None:
    """Run DeepEval metrics if available. Returns dict of scores or None."""
    try:
        from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
    except ImportError:
        return None

    if not os.environ.get("OPENAI_API_KEY"):
        return None

    context = [s.get("title", "") + ": " + s.get("content", "")[:500] for s in sources]
    context = [c for c in context if c.strip()]

    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=context if context else None,
    )

    scores = {}
    for metric_cls, name in [(FaithfulnessMetric, "faithfulness"), (AnswerRelevancyMetric, "relevancy")]:
        try:
            metric = metric_cls(threshold=0.5)
            metric.measure(test_case)
            scores[name] = metric.score
        except Exception:
            scores[name] = 0.0

    return scores


# ---- Main Runner ------------------------------------------------------------------------

def print_summary(total: int, passed: int, failed: int, deepeval_scores: dict[str, list[float]] | None, duration: float):
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total tests:  {total}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {failed}")
    print(f"Duration:     {duration:.1f}s")

    if deepeval_scores:
        print("\nDeepEval Metrics (average):")
        for metric, values in deepeval_scores.items():
            if values:
                avg = sum(values) / len(values)
                print(f"  {metric.capitalize()}: {avg:.2f}")

    print("=" * 60)
    if failed == 0:
        print("RESULT: ALL TESTS PASSED")
    else:
        print(f"RESULT: {failed} TEST(S) FAILED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run agentic RAG evaluation suite")
    parser.add_argument("--deepeval", action="store_true", help="Enable DeepEval metrics (requires OPENAI_API_KEY)")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max agent iterations per query")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed failures")
    args = parser.parse_args()

    # Load test cases
    golden_path = ROOT / "evals" / "golden.jsonl"
    if not golden_path.exists():
        print(f"ERROR: Golden dataset not found at {golden_path}")
        sys.exit(1)

    test_cases = load_test_cases(golden_path)
    print(f"Loaded {len(test_cases)} test cases from {golden_path}")
    print(f"DeepEval: {'enabled' if args.deepeval else 'disabled'}")

    # Run tests
    passed = 0
    failed = 0
    deepeval_scores: dict[str, list[float]] = {"faithfulness": [], "relevancy": []}
    start_time = time.time()

    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {tc.category} | {tc.query[:60]}...")
        sys.stdout.flush()

        try:
            result = run_agent(tc.query, max_iterations=args.max_iterations)
        except Exception as e:
            print(f"  [FAIL] AGENT ERROR: {e}")
            failed += 1
            continue

        ok, failures = tc.evaluate(result)

        if ok:
            tools_used = [s["tool"] for s in result["steps"]]
            print(f"  [PASS] tools={tools_used} sources={len(result['sources'])} iters={result['iterations']}")
            passed += 1
        else:
            print(f"  [FAIL]")
            for f in failures:
                print(f"    - {f}")
            if args.verbose:
                print(f"    Answer: {result['answer'][:200]}...")
            failed += 1

        # DeepEval (optional)
        if args.deepeval:
            scores = run_deepeval(tc.query, result["answer"], result.get("sources", []))
            if scores:
                for k, v in scores.items():
                    deepeval_scores[k].append(v)
                print(f"    DeepEval: faithfulness={scores.get('faithfulness', 0):.2f}, relevancy={scores.get('relevancy', 0):.2f}")

    duration = time.time() - start_time
    print_summary(len(test_cases), passed, failed, deepeval_scores if args.deepeval else None, duration)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()