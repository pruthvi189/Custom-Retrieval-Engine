"""Single FastAPI application that routes every /api/* path to the store layer.

Port of the Node handler (``handler.js`` + ``http.js``). One app keeps cold
starts (and deploys) simple instead of one serverless function per endpoint,
and it preserves the original HTTP contract exactly so ``index.html`` works
unchanged: same routes, same query params, same JSON shapes and error
strings, ``distance`` rounded to 6 decimals.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# Make engine.* / api.* importable on Vercel's Python runtime regardless of
# how it lays out the module path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Local dev: load provider keys from .env.local (Vercel provides them natively
# in the cloud, where this file does not exist). Must happen before api.store
# is imported - providers reads the environment at import time.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env.local")

from fastapi import FastAPI, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

import api.store as store  # noqa: E402
from engine import DIMS  # noqa: E402

app = FastAPI(redirect_slashes=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

_ID_RE = re.compile(r"^\d+$")
_INT_RE = re.compile(r"^\s*[+-]?\d+")


# ---- HTTP helpers (port of http.js) ----------------------------------------------------


def json_response(status: int, obj: Any) -> Response:
    return Response(
        content=json.dumps(obj, ensure_ascii=False, separators=(",", ":")),
        status_code=status,
        media_type="application/json",
    )


def round6(x) -> float:
    """Matches ``Number(h.distance.toFixed(6))`` in the original handler."""
    return round(float(x), 6)


def parse_v(s: str | None) -> list[float]:
    """Mirrors ``(query.v || '').split(',').map(Number).filter(Number.isFinite)``.

    Note ``Number('') === 0``, so a missing ``v`` parses to ``[0]``.
    """
    out = []
    if not s:
        out.append(0.0)
        return out
    for token in s.split(","):
        token = token.strip()
        if token == "":
            val = 0.0
        else:
            try:
                val = float(token)
            except ValueError:
                continue
        if math.isfinite(val):
            out.append(val)
    return out


def parse_int(s: str | None, default: int) -> int:
    """Mirrors ``parseInt(s, 10) || default`` (so ``k=0``, ``k=abc`` -> default)."""
    if s is None:
        return default
    m = _INT_RE.match(s)
    if not m:
        return default
    val = int(m.group(0), 10)
    return val if val else default


async def read_body(request: Request) -> dict:
    """Replicates ``typeof req.body === 'string' ? JSON.parse(req.body) : (req.body || {})``."""
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


# ---- route table (port of handler.js) --------------------------------------------------


@app.options("/{full_path:path}")
async def cors_preflight(full_path: str) -> Response:
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    return Response(status_code=204, headers=headers)


@app.get("/api/items")
def get_items() -> Response:
    return json_response(200, store.items())


@app.get("/api/search")
def search(
    v: str = Query(""),
    k: str = Query(""),
    metric: str = Query(""),
    algo: str = Query(""),
) -> Response:
    q = parse_v(v)
    if len(q) != DIMS:
        return json_response(400, {"error": "need 16D vector"})
    kk = parse_int(k, 5)
    mm = metric or "cosine"
    aa = algo or "hnsw"
    out = store.search_items(q, kk, mm, aa)
    return json_response(
        200,
        {
            "results": [
                {
                    "id": h["id"],
                    "metadata": h["metadata"],
                    "category": h["category"],
                    "distance": round6(h["distance"]),
                    "embedding": h["embedding"],
                }
                for h in out["hits"]
            ],
            "latencyUs": max(0, round(out["us"])),
            "algo": out["algo"],
            "metric": out["metric"],
        },
    )


@app.post("/api/insert")
async def insert(request: Request) -> Response:
    body = await read_body(request)
    meta = str(body.get("metadata", "")).strip()
    cat = str(body.get("category", "")).strip()
    emb_raw = body.get("embedding")
    emb = [float(x) for x in emb_raw] if isinstance(emb_raw, list) else []
    if not meta or len(emb) != DIMS:
        return json_response(400, {"error": "invalid body"})
    return json_response(200, {"id": store.insert_item(meta, cat, emb)})


@app.delete("/api/delete/{item_id}")
def delete_item(item_id: str) -> Response:
    if not _ID_RE.match(item_id):
        return json_response(404, {"error": "not found"})
    return json_response(200, {"ok": store.delete_item(int(item_id, 10))})


@app.get("/api/benchmark")
def benchmark(
    v: str = Query(""),
    k: str = Query(""),
    metric: str = Query(""),
) -> Response:
    q = parse_v(v)
    if len(q) != DIMS:
        return json_response(400, {"error": "need 16D vector"})
    b = store.benchmark(q, parse_int(k, 5), metric or "cosine")
    return json_response(
        200,
        {
            "bruteforceUs": round(b["bruteforceUs"]),
            "kdtreeUs": round(b["kdtreeUs"]),
            "hnswUs": round(b["hnswUs"]),
            "itemCount": b["itemCount"],
        },
    )


@app.get("/api/hnsw-info")
def hnsw_info() -> Response:
    return json_response(200, store.hnsw_info())


@app.get("/api/status")
def status() -> Response:
    return json_response(200, store.status())


@app.get("/api/stats")
def stats() -> Response:
    store.items()
    return json_response(200, store.stats())


@app.post("/api/agent/ingest")
async def agent_ingest(request: Request) -> Response:
    body = await read_body(request)
    topic = str(body.get("topic") or body.get("question") or "").strip()
    max_articles = parse_int(str(body.get("maxArticles", "")), 1)
    if not topic:
        return json_response(400, {"error": "need topic"})
    result = store.web_ingest(topic, max_articles)
    if result.get("error"):
        return json_response(400, {"error": result["error"]})
    return json_response(200, result)


@app.post("/api/doc/insert")
async def doc_insert(request: Request) -> Response:
    body = await read_body(request)
    title = str(body.get("title", "")).strip()
    text = str(body.get("text", "")).strip()
    if not title or not text:
        return json_response(400, {"error": "need title and text"})
    result = store.doc_insert(title, text)
    if result.get("error"):
        return json_response(400, {"error": result["error"]})
    return json_response(200, {"ids": result["ids"], "chunks": result["chunks"], "dims": result["dims"]})


@app.get("/api/doc/list")
def doc_list() -> Response:
    return json_response(200, store.doc_list())


@app.post("/api/doc/search")
async def doc_search(request: Request) -> Response:
    body = await read_body(request)
    question = str(body.get("question", "")).strip()
    k = parse_int(str(body.get("k", "")), 3)
    if not question:
        return json_response(400, {"error": "need question"})
    result = store.doc_search(question, k)
    if result.get("error"):
        return json_response(400, {"error": result["error"]})
    return json_response(200, {"contexts": result["contexts"]})


@app.post("/api/doc/ask")
async def doc_ask(request: Request) -> Response:
    body = await read_body(request)
    question = str(body.get("question", "")).strip()
    k = parse_int(str(body.get("k", "")), 3)
    if not question:
        return json_response(400, {"error": "need question"})
    result = store.doc_ask(question, k)
    if result.get("error"):
        return json_response(400, {"error": result["error"]})
    return json_response(200, result)


@app.delete("/api/doc/delete/{chunk_id}")
def doc_delete(chunk_id: str) -> Response:
    if not _ID_RE.match(chunk_id):
        return json_response(404, {"error": "not found"})
    return json_response(200, {"ok": store.doc_delete(int(chunk_id, 10))})


# ---- parity handlers (the places "idiomatic FastAPI" diverges) -------------------------


@app.middleware("http")
async def strip_trailing_slashes(request: Request, call_next):
    path = request.scope["path"]
    if path != "/" and path.endswith("/"):
        request.scope["path"] = path.rstrip("/")
    return await call_next(request)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    # Wrong method on a known path -> 404, not 405; unknown path -> 404 (both
    # fall through to `send(404, {error: 'not found'})` in the original).
    if exc.status_code in (404, 405):
        return json_response(404, {"error": "not found"})
    return json_response(exc.status_code, {"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return json_response(500, {"error": str(exc)})
