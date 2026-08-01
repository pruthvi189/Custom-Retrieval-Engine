# Custom Retrieval Engine

A vector database built from scratch — three search indexes, a document store,
and a grounded RAG pipeline, running as Vercel serverless functions behind a
single-page UI.

**Live demo: [https://customrag.vercel.app](https://customrag.vercel.app)**

I've always been curious about what actually goes on inside tools like Pinecone
or Weaviate, so instead of pulling in a library I wrote the whole thing myself.
Everything here is plain Node.js, small enough to read front to back.

## What it does

The demo page has three main pieces:

- **Vector search.** Three index implementations — HNSW, KD-tree, and brute
  force — stay in sync so you can search the same data with each and compare
  their timings side by side. Pick a metric (cosine, euclidean, manhattan) and
  a k.
- **The scatter plot.** A 2D projection of the 16D demo index: 20 pre-loaded
  vectors across CS, math, food, and sports, plus anything you add. Hover a dot
  to see what it is.
- **Documents + RAG.** Paste in notes, a blog post, whatever. The text gets
  chunked (250 words, 30 words of overlap), embedded via OpenRouter, and
  stored. Then you can ask questions and the model answers strictly from what
  you've stored — not from its own memory. If nothing matches, it says so
  instead of guessing.

There's also a "Search the web for more" button that grabs a Wikipedia article,
chunks it, and drops it into the document store — no extra API key required.

## How retrieval works

```
your question
  └─ OpenRouter text-embedding-3-small → 1536D vector
        └─ nearest-neighbour search (cosine, threshold 0.7)
              └─ top-k chunks → Groq llama-3.3-70b-versatile
                    └─ answer, grounded in those chunks only
```

The prompt tells the model to use nothing but the retrieved context. Empty
database, or nothing clearing the similarity threshold? You get
`"Not found in your documents."` and a `notFound` flag — no hallucinations.

## Project layout

```
api/
  handler.js          one serverless function that routes every /api/* path
  _lib/
    core.js           the engine: distance metrics, heaps, BruteForce,
                      KDTree, HNSW, VectorDB, DocumentDB, chunker, demo data
    store.js          persistence layer (in-memory, or Postgres + pgvector
                      when DATABASE_URL is set) plus RAG and web-ingest logic
    providers.js      OpenRouter embeddings + Groq generation clients
    http.js           small CORS / JSON helpers
index.html            the single-page UI
test/smoke.js         local smoke test
```

### The indexes

- **BruteForce** — the honest baseline: linear scan, sort, keep k.
- **KDTree** — axis-aligned binary space partitioning. Exact in low
  dimensions; search prunes whole subtrees when the splitting plane is farther
  away than the current k-th best. Deletes rebuild the tree.
- **HNSW** — a multilayer small-world graph. Each node gets a random max layer;
  higher layers are sparser and act like highways during search. Inserts run a
  beam search per layer (`ef_construction`) and keep neighbor lists bounded.

### Persistence

Storage defaults to in-memory, which on serverless means data can vanish when a
function goes cold. Set `DATABASE_URL` to a Postgres database with pgvector and
the store layer switches to real persistence: tables are created on boot, demo
vectors are seeded once, and the index reloads from the database.

## Running locally

```bash
npm install
vercel dev
```

The demo index works with no keys at all — search, benchmark, and the scatter
plot all run offline. You only need the provider keys for the Ask AI tab:

```
OPENROUTER_API_KEY   # embeddings
GROQ_API_KEY         # answer generation
DATABASE_URL         # optional: Postgres with pgvector
```

## API

Base path: `/api`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/items` | all vectors |
| GET | `/search?v=0.1,0.2,...&k=5&metric=cosine&algo=hnsw` | k-NN search |
| POST | `/insert` | insert a vector `{ metadata, category, embedding }` |
| DELETE | `/delete/:id` | delete a vector |
| GET | `/benchmark?v=...&k=5&metric=cosine` | time all three algorithms |
| GET | `/hnsw-info` | HNSW graph structure per layer |
| GET | `/status` | provider health + model info |
| POST | `/doc/insert` | embed and store a document `{ title, text }` |
| GET | `/doc/list` | list stored chunks |
| DELETE | `/doc/delete/:id` | delete a chunk |
| POST | `/doc/search` | retrieve matching chunks `{ question, k }` |
| POST | `/doc/ask` | grounded RAG answer `{ question, k }` |
| POST | `/agent/ingest` | fetch a Wikipedia article and store it `{ topic }` |

Example:

```bash
curl -X POST https://customrag.vercel.app/api/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

## Deploying

```bash
vercel deploy --prod
```

Then add `OPENROUTER_API_KEY` and `GROQ_API_KEY` (plus `DATABASE_URL` if you
want persistence) as Vercel environment variables and redeploy once.

---

*This is the Node port of an original C++ implementation of the same
algorithms. The C++ source stays out of this repository.*
