"""pg + pgvector persistence layer with an in-memory fallback.

If DATABASE_URL or POSTGRES_URL is set, vectors and doc chunks are persisted
to a pgvector database and reloaded on each cold start. Otherwise everything
runs in-memory (which is ephemeral on serverless).
"""

from __future__ import annotations

import os
import sys
import threading

import psycopg
from psycopg.rows import dict_row

# Kept as module constants so the schema can be asserted by tests (guard
# against drift from the original schema).
DDL_ITEMS = (
    "CREATE TABLE IF NOT EXISTS items "
    "(id SERIAL PRIMARY KEY, metadata TEXT NOT NULL, category TEXT NOT NULL, "
    "embedding vector(16) NOT NULL)"
)
DDL_DOCS = (
    "CREATE TABLE IF NOT EXISTS doc_chunks "
    "(id SERIAL PRIMARY KEY, title TEXT NOT NULL, text TEXT NOT NULL, "
    "embedding vector(1536) NOT NULL)"
)

_conn = None
_init_lock = threading.Lock()
_query_lock = threading.RLock()


def db_url() -> str:
    return os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or ""


def vec_to_string(v) -> str:
    """pgvector stores vectors as text like [0.12,0.90,...]"""
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


def get_pool():
    """Return a single psycopg connection or None (in-memory fallback).

    Single-flight init like the JS promise: create the extension + tables on
    first use; any failure logs and falls back to in-memory.
    """
    global _conn
    if _conn is not None:
        return _conn
    if not db_url():
        return None
    with _init_lock:
        if _conn is not None:
            return _conn
        try:
            conn = psycopg.connect(
                db_url(),
                connect_timeout=8,
                sslmode="require",
            )
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(DDL_ITEMS)
                cur.execute(DDL_DOCS)
            conn.commit()
            _conn = conn
        except Exception as e:  # noqa: BLE001 - mirrors JS fallback behavior
            print(f"pg init failed, falling back to in-memory: {e}", file=sys.stderr)
            _conn = None
    return _conn


def query(sql: str, params=None) -> list[dict]:
    """Run a statement on the shared connection under a lock.

    Returns rows as dicts; commits after every call (fine at this scale).
    Raises RuntimeError if no database is available - callers check
    ``get_pool()`` first.
    """
    conn = get_pool()
    if conn is None:
        raise RuntimeError("no database available")
    with _query_lock:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            try:
                rows = cur.fetchall()
            except psycopg.ProgrammingError:
                rows = []
            conn.commit()
    return rows
