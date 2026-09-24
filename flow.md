# Custom Retrieval Engine — Data Flow & Architecture

An advanced-README walkthrough of the whole system: every flow from start to finish, what runs where, and how the pieces fit. Accurate to the code in this repo.

- Live demo: https://customrag.vercel.app
- Repo: `pruthvi189/Custom-Retrieval-Engine` (branch `master`)

---

## 1. Big picture

```
  index.html (SPA, 4 tabs) ──/api/*──▶ api/index.py (FastAPI, one function)
                                          │
                      ┌────────────────────┼────────────────────┐
                      ▼                    ▼                    ▼
               api/store.py        api/agent.py          api/providers.py
               (persistence,       (ReAct agent loop,    (OpenRouter clients)
                RAG, wiki ingest)   background queue)
                                    api/tools.py
                      │
                      ▼
               engine/ (vector search core: heapq · scipy · numpy)
               ─────────────────────────────────────────────────
               distance.py · heaps.py · kdtree.py · hnsw.py
               chunking.py · text_features.py · demo.py · vectordb.py
               ─────────────────────────────────────────────────
                     │ optional
                     ▼
              PostgreSQL + pgvector (only if DATABASE_URL/POSTGRES_URL set)
```

Everything is served by one FastAPI app (`api/index.py`) running as a single Vercel serverless function. The frontend is one vanilla-JS `index.html`. The vector-search indexes live in `engine/` — HNSW hand-written, kd-tree on scipy (plus a hand-kept cosine path), heaps on `heapq` — no external vector database.

---

## 2. The two indexes

| Index | Dims | Data | Search | Persistence |
|---|---|---|---|---|
| `VectorDB` (`engine/vectordb.py`) | 16 | 20 demo vectors + doc graph points | HNSW · KD-Tree · Brute Force, kept in sync | `items` table |
| `DocumentDB` (`engine/vectordb.py`) | 1536 | text chunks from docs / Wikipedia | brute-force cosine only | `doc_chunks` table |

- `VectorDB` holds the 16D demo space used by the Search tab, the benchmark, and the PCA visualizer.
- `DocumentDB` holds real embedded text (OpenRouter, `text-embedding-3-small`). This is what Ask AI retrieves from.
- Both are in-memory on serverless; with a `DATABASE_URL` they reload from pgvector on every cold start.

---

## 3. The engine core (`engine/`)

- `distance.py` — `euclidean`, `cosine` (returned as `1 - similarity` so one code path handles every metric), `manhattan`, and `get_dist_fn`; numpy-backed, returning builtin `float`.
- `heaps.py` — `MinHeap`/`MaxHeap` on `(distance, id)`, backed by stdlib `heapq`.
- `kdtree.py` — scipy `cKDTree` for euclidean/manhattan k-NN, hand-written axis-cycling tree with hyperplane pruning for cosine; exact, best under ~20 dims; delete rebuilds the tree.
- `hnsw.py` — hierarchical navigable small-world graph (`M=16`, `ef_construction=200`, `ef_search=50`); approximate; deletion severs back-references; hand-written.
- `chunking.py` — `chunk_text(text, 250, 30)`: sliding 250-word windows with 30-word overlap (step 220).
- `text_features.py` — `graph_embedding(text)` → deterministic 16D vector from keyword hits across 4 buckets (cs/math/food/sports). No jitter server-side.
- `demo.py` — fixed 20-item demo corpus (cs/math/food/sports).
- `vectordb.py` — `Item` dataclass + `BruteForce` (the O(n) reference baseline) + `VectorDB` (16D, three sync indexes) + `DocumentDB` (1536D, brute force).

No external vector database anywhere — heaps come from stdlib `heapq`, euclidean/manhattan k-NN from scipy `cKDTree`, metrics from numpy, and HNSW is hand-written.

---

## 4. Persistence & cold starts (`api/db.py`, `api/store.py`)

- With `DATABASE_URL` or `POSTGRES_URL`: `get_pool()` lazily connects once, runs `CREATE EXTENSION IF NOT EXISTS vector`, and creates the tables:
  - `items (id SERIAL, metadata TEXT, category TEXT, embedding vector(16))`
  - `doc_chunks (id SERIAL, title TEXT, text TEXT, embedding vector(1536))`
- On every request path, `ensure_demo()` seeds the demo corpus once if empty and reloads the in-memory index from Postgres.
- The cached connection is dropped and recreated if Postgres kills it (serverless scale-to-zero) — `query()` retries once after a `psycopg.OperationalError`.
- Without a DB URL everything runs in-memory and resets on cold start (search, benchmark, visualization still work, key-free).

---

## 5. End-to-end flows

### 5.1 Demo-vector search (Search tab) — no API keys

```
text → textToEmbedding(text) [browser, 16D keyword embed]
     → GET /api/search?v=<16 floats>&k=<n>&metric=<cosine|euclidean|manhattan>&algo=<hnsw|kdtree|bruteforce>
     → api/index.py → store.search_items() → vdb.search()
     → top-k hits {id, metadata, category, distance, embedding} + latencyUs
     → render: top matches, query-embedding bar chart, HNSW layer inspector
```

- The browser-side `textToEmbedding` uses the same 4 keyword buckets as `engine/text_features.py` so manual inserts and fetched articles land in the same graph region.
- `Compare all algorithms` calls `GET /api/benchmark` which times all three indexes on the same query.
- `GET /api/hnsw-info` powers the per-layer node/edge graph inspector.

### 5.2 Manual document ingestion (Documents tab)

```
title + text → POST /api/doc/insert
     → chunk_text(text, 250, 30) → list of chunks
     → embed each chunk via OpenRouter (1536D)   [parallel]
     → store in doc_chunks (+ in-memory DocumentDB); chunk title = "Title [i/N]"
     → client adds a 16D graph point via textToEmbedding for the scatter plot
```

- `GET /api/doc/list` lists chunks with preview + word count.
- `DELETE /api/doc/delete/:id` deletes one chunk. (Known quirk: returns `{"ok":false}` even when the Postgres delete succeeds — verify via `/api/doc/list`.)

### 5.3 Ask AI — grounded RAG (Ask AI tab)

```
question → POST /api/doc/search {question, k}
     → embed question via OpenRouter (1536D)
     → brute-force cosine over doc_chunks, filter distance <= 0.7
     → top-k {id, title, distance}
     → frontend maps each chunk to its 16D doc graph point (longest-prefix match
       of title vs point metadata) → triangle on the top-1 hit's point
     → POST /api/doc/ask {question, k}
     → same retrieval, then a helpful-assistant prompt (mirrors the reference repo):
         LLM uses the retrieved chunks when relevant, otherwise falls back to
         its own general knowledge; it must not mention the context. It always
         answers — never a hard refusal. (notFound:true only when 0 chunks matched)
     → render answer + model name + clickable retrieved contexts
```

Rules that keep it honest:
- Empty DB or nothing above distance 0.7 → the LLM still answers from general
  knowledge; the UI offers "Search the web for more" to ingest sources.
- The triangle only ever sits on a retrieved chunk's own graph point.

### 5.4 Agentic research (Agent tab)

```
question → POST /api/agent/ask {query, maxIterations}
     → agent.run_agent() — ReAct loop, max 5 iterations:

       1. planner LLM call → JSON {tool, input, reason}
          tool ∈ doc_search | wiki_search | web_search | finish
          (parse failure → retry once → fall back to web_search)
       2. execute tool → normalized ToolResult {tool, query, results, sources, error?}
       3. accumulate observations/sources/context
       4. loop until finish or max iterations

     → reflection step: LLM answers YES/NO "enough evidence?"; if NO, one extra
       web_search pass
     → final synthesis: grounded prompt, citations like [Source: Wikipedia - X]
     → returns {answer, iterations, steps[], sources, context}
```

Tools:
- `doc_search` — semantic search of the local knowledge base.
- `wiki_search` — Wikipedia fetch + **permanent ingest** (MediaWiki API, no key; chunk → parallel embed → store, capped at 10 chunks/article → deduped 16D graph point). Can run as a background thread task.
- `web_search` — Tavily live-web search (needs `TAVILY_API_KEY`), **temporary** context only, never persisted.

`💾 Save to Knowledge Base` on an agent answer calls `/api/doc/insert` (and adds a 16D point), so a synthesized answer becomes retrievable by Ask AI.

### 5.5 Wikipedia ingestion directly (no agent)

```
POST /api/agent/ingest {topic, maxArticles}
     → store.web_ingest(topic, max_articles)
     → wiki_search: MediaWiki search → top titles
     → wiki_fetch: plain-text extract per title (explaintext=1)
     → chunk → parallel embed → store (cap 10 chunks/article)
     → _ensure_graph_point(title): 16D point, deduped per unique title
     → {added:[{title, chunks, stored}], dims, message}
```

---

## 6. HTTP contract notes (`api/index.py`)

The FastAPI app is a byte-for-byte port of the original Node handler. Details that matter:

- `distance` rounded to 6 decimals (`round6`).
- Missing `v` parses to `[0]` (mirrors `Number('') === 0`).
- `k=0` or `k=abc` → default (mirrors `parseInt(s, 10) || default`).
- Trailing slashes stripped by middleware.
- Wrong method on a known path → `404 {error:"not found"}`, not 405.
- CORS wide open (`*`); methods GET/POST/DELETE/OPTIONS.

---

## 7. Providers (`api/providers.py`)

- `embed_one(text)` — OpenRouter `/api/v1/embeddings`, `openai/text-embedding-3-small`, 1536D.
- `generate(prompt)` — OpenRouter `/api/v1/chat/completions`, `meta-llama/llama-3.3-70b-instruct`, `max_tokens=1000`, `temperature=0.7`. Returns the text or an `ERROR: ...` string.
- Keys from env vars only. Legacy naming: `GROQ_API_KEY` / `groqAvailable` — generation actually goes through OpenRouter now.
- Optional `TAVILY_API_KEY` for agent live web search.

---

## 8. Frontend internals (`index.html`)

- Four tabs: Search, Documents, Ask AI, Agent. Module globals: `allItems`, `pcaPoints`, `hitIds`, `queryPt`.
- `textToEmbedding` — browser-side 16D keyword embedder (mirrors `text_features.py`, plus render jitter).
- `pca2D` — in-browser PCA (power iteration, 200 iters, Gram–Schmidt for PC2) → 2D scatter on Canvas, color-coded by category.
- `askAI()` — retrieval → graph-point mapping → triangle on top-1 → grounded answer (see 5.3).
- Agent tab renders each iteration step as it runs, then the final answer.

---

## 9. Running & deploying

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python scripts/run_server.py      # http://127.0.0.1:8000
.venv/Scripts/python -m pytest -q               # ~106 tests, no keys needed
.venv/Scripts/python scripts/parity_check.py    # contract check
vercel deploy --prod                            # deploy
```

Env vars: `OPENROUTER_API_KEY`, `GROQ_API_KEY` (legacy), `DATABASE_URL` (optional), `TAVILY_API_KEY` (optional). `vercel.json` rewrites `/api/*` → `api/index`, `maxDuration: 60`.

---

## 10. Known quirks / gotchas

- `DELETE /api/doc/delete/:id` returns `{"ok":false}` even on success (returns the in-memory `DocumentDB.remove()` result, which is False when chunks live only in Postgres). The DB delete happens; confirm via `/api/doc/list`.
- `groqAvailable` / `GROQ_API_KEY` are legacy names — both embeddings and generation go through OpenRouter.
- The 16D keyword embedding is for the visualizer only, not semantic ranking. Real RAG ranking uses 1536D OpenRouter embeddings + brute-force cosine.
- Document retrieval is O(n); fine for demo scale, not for large corpora.
- Background ingestion uses an in-process thread queue — on Vercel this would be replaced by Redis/SQS in production.
