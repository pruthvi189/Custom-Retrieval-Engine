# Evaluation Suite

Lightweight evaluation framework for the agentic RAG system.

## Why Evaluations?

- **Regression detection** — Catch broken agent loops, missing sources, empty answers
- **CI/CD gate** — Block merges that degrade basic functionality
- **Confidence** — Know the agent runs end-to-end before deploying

## Quick Start

```bash
# Basic evaluation (no extra deps)
python -m evals.runner

# With DeepEval metrics (requires OPENAI_API_KEY)
pip install deepeval
export OPENAI_API_KEY=sk-...
python -m evals.runner --deepeval
```

Exit code is `0` if all tests pass, `1` otherwise — CI friendly.

## Test Cases

Defined in `evals/golden.jsonl` — one JSON object per line:

```json
{
  "query": "What is HNSW?",
  "category": "document_retrieval"
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `query` | Yes | User question to test |
| `category` | No | Grouping for reporting (default: "uncategorized") |

### Categories

- `document_retrieval` — Factual questions answerable from docs/Wikipedia
- `web_search` — Current events, recent developments
- `multi_tool` — Requires combining multiple source types
- `reasoning` — Why/how questions about agent behavior
- `edge_case` — Empty results, nonsense queries, very long queries

## Adding a New Test

1. Open `evals/golden.jsonl`
2. Append one line: `{"query": "Your question", "category": "category_name"}`
3. Run `python -m evals.runner` to verify

## DeepEval Metrics (Optional)

When enabled with `--deepeval`:

| Metric | What It Measures |
|--------|------------------|
| **Faithfulness** | Is the answer grounded in the retrieved sources? |
| **Answer Relevancy** | Does the answer actually address the question? |

Requires `OPENAI_API_KEY` (used by DeepEval for LLM-as-judge).

## CI Integration

```yaml
# .github/workflows/eval.yml
- name: Run Evaluations
  run: python -m evals.runner --max-iterations 3
  env:
    GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
    TAVILY_API_KEY: ${{ secrets.TAVILY_API_KEY }}
```

Add `--deepeval` and `OPENAI_API_KEY` for LLM-based metrics in CI.

## Output Example

```
Loaded 15 test cases from .../evals/golden.jsonl
DeepEval: disabled

[1/15] document_retrieval | What is a binary tree?...
  [PASS] tools=['web_search', 'wiki_search'] sources=12 iters=3

[15/15] reasoning | How does reflection improve agent accuracy?...
  [PASS] tools=['web_search', 'web_search', 'web_search'] sources=18 iters=3

============================================================
EVALUATION SUMMARY
============================================================
Total tests:  15
Passed:       15
Failed:       0
Duration:     113.4s
============================================================
RESULT: ALL TESTS PASSED
============================================================
```

## What Gets Checked

Each test verifies the agent:
1. Returns a non-empty answer
2. Executes at least one step
3. Returns at least one source
4. Completes at least one iteration

## Design Principles

- **No abstractions** — Direct, readable code (~200 lines)
- **Reuses agent API** — Calls `api.agent.run_agent()` directly
- **Single file runner** — Easy to audit and modify
- **JSONL format** — Human-editable, diff-friendly, streaming parse
- **Optional DeepEval** — Core evals work without extra deps
- **Fast by default** — `--max-iterations 3` runs in ~2 minutes