# Custom Retrieval Engine

A vector retrieval engine built **from scratch** — HNSW, KD-Tree, and brute-force search implemented by hand, backed by a chunked document store and a grounded RAG pipeline. Everything runs as Vercel serverless functions behind a single-page UI.

I've always wondered what actually lives inside tools like Pinecone or Weaviate. So instead of importing one, I wrote the whole thing — distance metrics, heaps, the indexes, the chunker, the RAG glue — in plain Node.js, small enough to read front to back.

> [!IMPORTANT]
> Unlike most RAG projects, this does **not** rely on an existing vector database for the core engine. The three search indexes — **HNSW**, **KD-Tree**, and **Brute Force** — are implemented directly in `api/_lib/core.js`, and a live benchmark lets you time all three against the same query.

---

## Demo

| | |
|---|---|
| **Live demo** | [https://customrag.vercel.app](https://customrag.vercel.app) |
| **Source** | [github.com/pruthvi189/Custom-Retrieval-Engine](https://github.com/pruthvi189/Custom-Retrieval-Engine) |

The demo needs **zero API keys** for vector search, benchmarking, and the visualization. Keys are only required for the document/RAG features.

---

## Key Features

- **Custom HNSW** — hierarchical navigable small-world graph (`M=16`, `ef_construction=200`, `ef_search=50`), built and maintained in code.
- **Custom KD-Tree** — axis-aligned binary space partitioning with hyperplane pruning during k-NN.
- **Brute Force baseline** — the honest O(n) reference every other index is compared against.
- **Live side-by-side benchmarking** — one query, all three algorithms, microsecond timings.
- **Document Store** — arbitrary text chunked (250 words / 30 overlap) and embedded at 1536D.
- **Grounded RAG** — the LLM only ever sees retrieved chunks; it answers strictly from them or says `Not found in your documents.`
- **Agentic document ingestion** — hand it a topic and it researches, chunks, embeds, and stores the knowledge automatically.
- **Wikipedia ingestion** — no API key required, via the MediaWiki API.
- **Multiple similarity metrics** — euclidean, cosine, and manhattan.
- **In-browser PCA visualizer** — a live 2D projection of the vector space, color-coded by category.
- **HNSW graph inspector** — per-layer node/edge structure rendered from `/api/hnsw-info`.
- **Optional PostgreSQL + pgvector persistence** — survives serverless cold starts via `DATABASE_URL`.
- **Serverless deployment** — one Vercel function routing every `/api/*` endpoint.

---

## Architecture

```
                    ┌────────────────────────────┐
                    │         User / UI          │
                    └─────────────┬──────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
   ◀ INGESTION (write path)              RETRIEVAL (read path) ▶
              │                                       │
  ┌───────────┴────────────┐        ┌─────────────────┴────────────────┐
  │ Manual: title + text   │        │ Query embedding                  │
  │ Agent: Wikipedia topic │        │ (OpenRouter, 1536D)              │
  └───────────┬────────────┘        └─────────────────┬────────────────┘
              │                                       │
              ▼                                       ▼
  ┌─────────────────────────┐        ┌──────────────────────────────────┐
  │ Chunking                │        │ Nearest-neighbour search         │
  │ (250 words, 30 overlap) │        │ (cosine, distance ≤ 0.7)         │
  └───────────┬─────────────┘        └─────────────────┬────────────────┘
              │                                       │
              ▼                                       │
  ┌─────────────────────────┐        ┌────────────────▼────────────────┐
  │ Embed each chunk        │───────▶│        Custom Retrieval Engine   │
  │ (OpenRouter, 1536D)     │        │                                  │
  └───────────┬─────────────┘        │  ┌──────────┬─────────┬────────┐ │
              │                      │  │   HNSW   │ KD-Tree │  Brute │ │
              ▼                      │  │  (graph) │  (tree) │ (scan) │ │
  ┌─────────────────────────┐        │  └──────────┴─────────┴────────┘ │
  │ Postgres + pgvector     │        │                                  │
  │ (optional persistence)  │◀──────▶└────────────────┬─────────────────┘
  └─────────────────────────┘                         │
                                                      ▼
                                  ┌──────────────────────────────────┐
                                  │ Top-k context chunks              │
                                  └────────────────┬─────────────────┘
                                                   │
                                                   ▼
                                  ┌──────────────────────────────────┐
                                  │ Groq (llama-3.3-70b-versatile)    │
                                  │ Grounded generation — context     │
                                  │ only, no external knowledge       │
                                  └────────────────┬─────────────────┘
                                                   │
                                                   ▼
                                  ┌──────────────────────────────────┐
                                  │ Final answer  /  "Not found in    │
                                  │                 your documents."  │
                                  └──────────────────────────────────┘
```

Both paths share one engine: the 16D demo index keeps all three algorithms in sync in a single `VectorDB`; document retrieval is a separate 1536D brute-force store.

---

## Retrieval Pipeline

```
text → chunk → embed → index → retrieve → ground → generate
```

1. **Chunking** — text is split into sliding windows of 250 words with 30 words of overlap (step 220), so boundary context survives chunk edges.
2. **Embedding** — every chunk is embedded with OpenRouter (`text-embedding-3-small`, 1536D).
3. **Indexing** — chunk vectors go into the document store (brute-force, fine at this scale). The 16D demo index is kept in sync across HNSW, KD-Tree, and brute force.
4. **Retrieval** — the query is embedded the same way; cosine distance picks the top-k chunks, filtered by a similarity threshold of `0.7`.
5. **Grounding** — only chunks that actually matched are handed to the model, each labeled with its source title.
6. **Generation** — Groq (`llama-3.3-70b-versatile`) answers from that context alone. Empty database or nothing above threshold → `Not found in your documents.` with a `notFound: true` flag. No guessing.

> [!NOTE]
> The same distance functions back every path — `euclidean`, `cosine` (returned as `1 − similarity` so one code path handles all metrics), and `manhattan`.

---

## Search Algorithms

### HNSW
**Why:** approximate nearest neighbours at scale; the workhorse of production vector search.
**Complexity:** search ~O(log n); insert ~O(log n · ef).
**Advantages:** sub-linear search, great recall/speed trade-off, incremental inserts.
**Disadvantages:** approximate (can miss true k-NN), memory-hungry graph, index quality depends on build parameters.

### KD-Tree
**Why:** exact search that prunes whole subtrees instead of scanning everything.
**Complexity:** search O(log n) average in low dimensions, O(n) worst case; delete rebuilds the tree O(n log n).
**Advantages:** exact results, fast in low dimensions.
**Disadvantages:** degrades sharply past ~20–30 dimensions; insert order affects balance.

### Brute Force
**Why:** the reference point — a full linear scan costs zero index structure.
**Complexity:** search O(n), memory O(n·d).
**Advantages:** exact, trivially correct, supports inserts/deletes instantly.
**Disadvantages:** O(n) per query; not viable as data grows.

| Algorithm | Typical search | Exact? | Memory | Notes |
|---|---|---|---|---|
| Brute Force | O(n) | yes | O(n·d) | baseline, always correct |
| KD-Tree | O(log n) (low-d) | yes | O(n·d) | best below ~20 dims |
| HNSW | O(log n) | approx. | O(n·M) | scales to big data |

---

## Agentic Capabilities

`POST /api/agent/ingest` turns a bare topic into stored knowledge:

1. **Research** — a topic like `"quantum computing"` is searched on Wikipedia and the top article fetched.
2. **Capture** — the article's plain-text extract is pulled down (no API key).
3. **Chunk + embed** — the text is chunked and up to 10 chunks are embedded via OpenRouter.
4. **Store** — chunks are persisted, and a 16D point is added to the visualizer graph so fetched knowledge appears alongside manually inserted documents.
5. **Dedupe** — re-ingesting the same topic doesn't duplicate the graph point.

This is more than a RAG chatbot: you never paste or curate the corpus yourself, and retrieval stays **grounded** because the answer is still generated exclusively from whatever the ingestion stored. Ingest is a capability the system performs on your behalf — not a manual data-loading step.

---

## Technical Challenges

- **Implementing HNSW** — layered graph construction, bounded neighbor selection, `ef` tuning across insert vs. search, and safe deletions that sever back-references.
- **Balancing the KD-Tree** — naive insertion can skew the tree; search quality depends on split choice and dimension cycling.
- **Chunk overlap tuning** — too little overlap loses context at boundaries; too much wastes embedding budget.
- **Similarity thresholds** — a 0.7 distance cutoff keeps irrelevant chunks out of the prompt without over-filtering.
- **Serverless persistence** — serverless functions are stateless and ephemeral; persistence required Postgres + pgvector with on-boot schema creation, one-time seeding, and index reload per cold start.
- **Grounded generation** — prompt engineering plus hard retrieval filters to keep the model from hallucinating or leaking general knowledge.

---

## Benchmarks

Measured live on Vercel: 22 items, 16D vectors, cosine distance, `k=5` (three runs of `/api/benchmark`).

| Run | Brute Force | KD-Tree | HNSW |
|---|---|---|---|
| 1 | 136 µs | 157 µs | 153 µs |
| 2 | 52 µs | 67 µs | 82 µs |
| 3 | 17 µs | 31 µs | 109 µs |

> [!TIP]
> At this dataset size all three are sub-millisecond and **overhead dominates** — these numbers are illustrative, not a verdict. The point of the benchmark endpoint is that it exists: feed it more data and watch how each index scales. Document search (1536D) is intentionally brute-force at this scale.

---

## Project Structure

```
api/
  handler.js          one serverless function routing every /api/* path
  _lib/
    core.js           the engine: distance metrics, heaps, BruteForce,
                      KDTree, HNSW, VectorDB, DocumentDB, chunker, demo data
    store.js          persistence (in-memory, or Postgres + pgvector when
                      DATABASE_URL is set), RAG, and agent ingestion logic
    providers.js      OpenRouter embeddings + Groq generation clients
    http.js           small CORS / JSON helpers
index.html            the single-page UI (search, benchmark, PCA plot, RAG)
test/
  smoke.js            local smoke test (npm test)
vercel.json           rewrites /api/* to the handler
```

---

## API

Base path: `/api`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/items` | all vectors in the demo index |
| GET | `/search?v=…&k=5&metric=cosine&algo=hnsw` | k-NN search (`algo`: `hnsw` \| `kdtree` \| `bruteforce`) |
| POST | `/insert` | insert a vector `{ metadata, category, embedding }` |
| DELETE | `/delete/:id` | delete a vector |
| GET | `/benchmark?v=…&k=5&metric=cosine` | time all three algorithms on one query |
| GET | `/hnsw-info` | HNSW graph structure per layer |
| GET | `/status` | provider health + model info |
| GET | `/stats` | index counts + available algorithms/metrics |
| POST | `/doc/insert` | embed and store a document `{ title, text }` |
| GET | `/doc/list` | list stored chunks |
| DELETE | `/doc/delete/:id` | delete a chunk |
| POST | `/doc/search` | retrieve matching chunks `{ question, k }` |
| POST | `/doc/ask` | grounded RAG answer `{ question, k }` |
| POST | `/agent/ingest` | fetch a Wikipedia article and store it `{ topic }` |

```bash
curl -X POST https://customrag.vercel.app/api/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Running Locally

```bash
npm install
vercel dev
```

Vector search, benchmark, and the PCA plot work with **no keys at all**. You only need provider keys for the Ask AI / document features:

```
OPENROUTER_API_KEY   # embeddings
GROQ_API_KEY         # answer generation
DATABASE_URL         # optional: Postgres with pgvector
```

Smoke test (with `vercel dev` running):

```bash
npm test
```

---

## Deployment

```bash
vercel deploy --prod
```

Then set `OPENROUTER_API_KEY` and `GROQ_API_KEY` as Vercel environment variables (plus `DATABASE_URL` for persistence) and redeploy once. Storage falls back to in-memory if `DATABASE_URL` is absent — everything still works, it just resets on cold starts.

---

## Future Improvements

- **Hybrid search** — combine vector similarity with keyword/BM25 signals.
- **IVF** — inverted-file clustering for faster approximate search.
- **Product Quantization** — compress vectors to shrink memory and speed scans.
- **DiskANN** — graph-based search designed for disk-resident datasets.
- **Metadata filtering** — restrict search to categories/fields before similarity.
- **Streaming ingestion** — incremental chunk + embed as documents grow.
- **Distributed indexing** — shard the graph across functions/instances.

---

*This is the Node port of an original C++ implementation of the same algorithms. The C++ source stays out of this repository.*
