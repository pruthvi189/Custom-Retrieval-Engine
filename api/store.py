"""Store layer - the in-memory engine with optional Postgres persistence.

Port of the original Node ``api/_lib/store.js``. If ``DATABASE_URL`` or
``POSTGRES_URL`` is set, vectors and doc chunks are persisted to a pgvector
database and reloaded on each cold start. Otherwise everything runs
in-memory (which is ephemeral on serverless).
"""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import httpx

import api.db as db
import api.providers as providers
from engine import DIMS, DEMO, DocumentDB, VectorDB, chunk_text, graph_embedding, load_demo

_STATE_LOCK = threading.RLock()

vdb = VectorDB(DIMS)
doc_db = DocumentDB()

_WIKI_UA = "CustomRetrievalEngine-RAG/1.0"


def db_active() -> bool:
    return db.get_pool() is not None


def _load_items() -> None:
    """Reload the in-memory index from Postgres (cold-start consistency)."""
    if not db_active():
        return
    rows = db.query(
        "SELECT id, metadata, category, embedding::text AS e FROM items ORDER BY id"
    )
    with _STATE_LOCK:
        vdb.reset()
        for row in rows:
            vdb.insert_raw(
                {
                    "id": row["id"],
                    "metadata": row["metadata"],
                    "category": row["category"],
                    "embedding": json.loads(row["e"]),
                }
            )


def _seed_pg() -> None:
    rows = db.query("SELECT COUNT(*)::int AS n FROM items")
    if rows and rows[0]["n"] == 0:
        for meta, cat, emb in DEMO:
            db.query(
                "INSERT INTO items (metadata, category, embedding) VALUES (%s, %s, %s::vector)",
                [meta, cat, db.vec_to_string(emb)],
            )


def ensure_demo() -> None:
    """Make sure the demo index has data before the first request."""
    if db_active():
        _seed_pg()
        _load_items()
    else:
        with _STATE_LOCK:
            if vdb.size() == 0:
                load_demo(vdb)


# ---- items / search / insert / delete -------------------------------------------------


def items() -> list[dict]:
    ensure_demo()
    return [item.vector_dict() for item in vdb.all()]


def search_items(q, k: int, metric: str, algo: str) -> dict:
    ensure_demo()
    _load_items()
    with _STATE_LOCK:
        return vdb.search(q, k, metric, algo)


def insert_item(meta: str, cat: str, emb) -> int:
    ensure_demo()
    if db_active():
        row = db.query(
            "INSERT INTO items (metadata, category, embedding) VALUES (%s, %s, %s::vector) RETURNING id",
            [meta, cat, db.vec_to_string(emb)],
        )
        id = row[0]["id"]
        with _STATE_LOCK:
            vdb.insert_raw({"id": id, "metadata": meta, "category": cat, "embedding": emb})
        return id
    with _STATE_LOCK:
        return vdb.insert(meta, cat, emb)


def delete_item(id: int) -> bool:
    ensure_demo()
    if db_active():
        db.query("DELETE FROM items WHERE id = %s", [id])
    with _STATE_LOCK:
        return vdb.remove(id)


def benchmark(q, k: int, metric: str) -> dict:
    ensure_demo()
    _load_items()
    with _STATE_LOCK:
        return vdb.benchmark(q, k, metric)


def hnsw_info() -> dict:
    ensure_demo()
    _load_items()
    with _STATE_LOCK:
        return vdb.hnsw_info()


# ---- documents + RAG -------------------------------------------------------------------


def store_chunk(chunk_title: str, text: str, emb) -> int:
    if db_active():
        row = db.query(
            "INSERT INTO doc_chunks (title, text, embedding) VALUES (%s, %s, %s::vector) RETURNING id",
            [chunk_title, text, db.vec_to_string(emb)],
        )
        return row[0]["id"]
    return doc_db.insert(chunk_title, text, emb)


def doc_insert(title: str, text: str) -> dict:
    ensure_demo()
    chunks = chunk_text(text, 250, 30)
    ids = []
    for i, chunk in enumerate(chunks):
        emb = providers.embed_one(chunk)
        if not emb:
            return {"error": "Embedding failed via OpenRouter. Check the OPENROUTER_API_KEY env var."}
        chunk_title = f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(store_chunk(chunk_title, chunk, emb))
    dims = 1536 if db_active() else doc_db.get_dims()
    return {"ids": ids, "chunks": len(chunks), "dims": dims}


def doc_list() -> list[dict]:
    ensure_demo()
    if db_active():
        rows = db.query("SELECT id, title, text FROM doc_chunks ORDER BY id")
    else:
        rows = [{"id": d.id, "title": d.title, "text": d.text} for d in doc_db.all()]
    out = []
    for d in rows:
        text = d["text"] or ""
        preview = text[:120] + "\u2026" if len(text) > 120 else text
        words = len([w for w in text.split() if w])
        out.append({"id": d["id"], "title": d["title"], "preview": preview, "words": words})
    return out


def doc_delete(id: int) -> bool:
    ensure_demo()
    if db_active():
        db.query("DELETE FROM doc_chunks WHERE id = %s", [id])
    return doc_db.remove(id)


def doc_search(question: str, k: int) -> dict:
    ensure_demo()
    q_emb = providers.embed_one(question)
    if not q_emb:
        return {"error": "Embedding failed via OpenRouter. Check the OPENROUTER_API_KEY env var."}
    if db_active():
        rows = db.query(
            "SELECT id, title, embedding <=> %s::vector AS distance "
            "FROM doc_chunks ORDER BY embedding <=> %s::vector LIMIT %s",
            [db.vec_to_string(q_emb), db.vec_to_string(q_emb), k],
        )
        contexts = [
            {"id": r["id"], "title": r["title"], "distance": float(r["distance"])}
            for r in rows
            if float(r["distance"]) <= 0.7
        ]
        return {"contexts": contexts}
    hits = doc_db.search(q_emb, k, 0.7)
    return {
        "contexts": [
            {"id": h["item"].id, "title": h["item"].title, "distance": h["distance"]}
            for h in hits
        ]
    }


def doc_ask(question: str, k: int) -> dict:
    ensure_demo()
    q_emb = providers.embed_one(question)
    if not q_emb:
        return {"error": "Embedding failed via OpenRouter. Check the OPENROUTER_API_KEY env var."}

    if db_active():
        doc_count = db.query("SELECT COUNT(*)::int AS n FROM doc_chunks")[0]["n"]
        rows = db.query(
            "SELECT id, title, text, embedding <=> %s::vector AS distance "
            "FROM doc_chunks ORDER BY embedding <=> %s::vector LIMIT %s",
            [db.vec_to_string(q_emb), db.vec_to_string(q_emb), k],
        )
        hits = [
            {
                "id": r["id"],
                "title": r["title"],
                "text": r["text"],
                "distance": float(r["distance"]),
            }
            for r in rows
            if float(r["distance"]) <= 0.7
        ]
    else:
        doc_count = doc_db.size()
        hits = [
            {
                "id": h["item"].id,
                "title": h["item"].title,
                "text": h["item"].text,
                "distance": h["distance"],
            }
            for h in doc_db.search(q_emb, k, 0.7)
        ]

    if doc_count == 0:
        return {
            "answer": 'No documents in the database yet. Use "Search the web for more" below to fetch knowledge about this topic.',
            "model": providers.GEN_MODEL,
            "contexts": [],
            "docCount": doc_count,
            "notFound": True,
        }

    if not hits:
        return {
            "answer": "Not found in your documents.",
            "model": providers.GEN_MODEL,
            "contexts": [],
            "docCount": doc_count,
            "notFound": True,
        }

    ctx = "".join(f"[{i + 1}] {h['title']}:\n{h['text']}\n\n" for i, h in enumerate(hits))
    prompt = (
        "You are a RAG assistant grounded strictly in the provided context. "
        "Answer the user's question using ONLY the context below. "
        "If the context does not contain enough information to answer, reply exactly: Not found in your documents. "
        "Do not use your own general knowledge. Do not mention the context.\n\n"
        "Context:\n" + ctx +
        "Question: " + question + "\n\n"
        "Answer:"
    )
    answer = providers.generate(prompt)
    not_found = bool(re.search(r"not found in your documents", answer, re.IGNORECASE))
    return {
        "answer": answer,
        "model": providers.GEN_MODEL,
        "contexts": hits,
        "docCount": doc_count,
        "notFound": not_found,
    }


# ---- web ingest (Wikipedia, no API key) ------------------------------------------------


def wiki_search(topic: str) -> list[str]:
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch="
        + quote(topic)
        + "&format=json&srlimit=3"
    )
    r = httpx.get(url, headers={"User-Agent": _WIKI_UA}, timeout=15.0)
    d = r.json()
    results = d.get("query", {}).get("search", []) if isinstance(d, dict) else []
    return [s.get("title") for s in results if isinstance(s, dict)]


def wiki_fetch(title: str) -> dict | None:
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles="
        + quote(title)
    )
    r = httpx.get(url, headers={"User-Agent": _WIKI_UA}, timeout=15.0)
    d = r.json()
    pages = d.get("query", {}).get("pages", {}) if isinstance(d, dict) else {}
    for pg in pages.values():
        extract = pg.get("extract") or ""
        if extract.strip():
            return {"title": pg.get("title"), "text": extract}
    return None


def graph_point_exists(meta: str) -> bool:
    if db_active():
        rows = db.query(
            "SELECT 1 FROM items WHERE metadata = %s AND category = 'doc' LIMIT 1",
            [meta],
        )
        return len(rows) > 0
    return any(i.metadata == meta and i.category == "doc" for i in vdb.all())


def web_ingest(topic: str, max_articles: int = 1) -> dict:
    ensure_demo()
    titles = wiki_search(topic)
    if not titles:
        return {"error": f'No Wikipedia results for "{topic}".'}
    added = []
    for title in titles[:max_articles]:
        art = wiki_fetch(title)
        if not art:
            continue
        chunks = chunk_text(art["text"], 250, 30)
        cap = chunks[:10]
        with ThreadPoolExecutor() as ex:
            embs = list(ex.map(providers.embed_one, cap))
        ids = []
        for i, chunk in enumerate(cap):
            emb = embs[i]
            if not emb:
                continue
            chunk_title = f"{art['title']} [{i + 1}/{len(cap)}]" if len(cap) > 1 else art["title"]
            ids.append(store_chunk(chunk_title, chunk, emb))
        # Give the article a 16D point on the visualizer graph so fetched
        # knowledge shows up next to manually inserted documents.
        if not graph_point_exists(art["title"]):
            insert_item(art["title"], "doc", graph_embedding(art["title"]))
        added.append(
            {
                "title": art["title"],
                "chunks": len(cap),
                "stored": len(ids),
                "truncated": len(chunks) > len(cap),
            }
        )
    if not added:
        return {"error": "Could not fetch article content."}
    dims = 1536 if db_active() else doc_db.get_dims()
    total = sum(a["stored"] for a in added)
    return {"added": added, "dims": dims, "message": f"Stored {total} chunk(s) from the web."}


# ---- status / stats --------------------------------------------------------------------


def status() -> dict:
    ensure_demo()
    if db_active():
        doc_count = db.query("SELECT COUNT(*)::int AS n FROM doc_chunks")[0]["n"]
        doc_dims = 1536 if doc_count > 0 else 0
    else:
        doc_count = doc_db.size()
        doc_dims = doc_db.get_dims()
    emb_up = providers.embed_available()
    groq_up = providers.groq_available()
    return {
        "groqAvailable": groq_up,
        "apiKeySet": bool(os.environ.get("GROQ_API_KEY")),
        "embedAvailable": emb_up,
        "embedKeySet": bool(os.environ.get("OPENROUTER_API_KEY")),
        "embedModel": providers.EMBED_MODEL,
        "genModel": providers.GEN_MODEL,
        "docCount": doc_count,
        "docDims": doc_dims,
        "demoDims": DIMS,
        "demoCount": vdb.size(),
    }


def stats() -> dict:
    return {
        "count": vdb.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    }
