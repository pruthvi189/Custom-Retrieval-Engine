import pytest

import engine
import api.db as db
import api.providers as providers
import api.store as store
from engine import DIMS

# Snapshot before the autouse fixture swaps it out, so opt-in Postgres
# integration tests can restore the real connection factory.
REAL_GET_POOL = db.get_pool


def fake_embed(text: str) -> list[float]:
    """Deterministic, content-dependent 1536D vector so no network is hit.

    Same text -> identical vector (cosine distance 0); different text ->
    orthogonal vector (cosine distance 1).  This lets RAG tests hit the
    0.7 threshold reliably in both directions.
    """
    v = [0.0] * 1536
    seed = sum(ord(ch) for ch in text)
    v[seed % 1536] = 1.0
    return v


def fake_generate(prompt: str) -> str:
    return "Paris is the capital of France."


@pytest.fixture(autouse=True)
def isolated_backends(monkeypatch):
    """Keep every test off Postgres and off the real provider APIs."""
    monkeypatch.setattr(db, "get_pool", lambda: None)
    monkeypatch.setattr(providers, "embed_available", lambda: False)
    monkeypatch.setattr(providers, "groq_available", lambda: False)
    monkeypatch.setattr(providers, "embed_one", fake_embed)
    monkeypatch.setattr(providers, "generate", fake_generate)


@pytest.fixture
def fresh_store():
    store.vdb = engine.VectorDB(DIMS)
    store.doc_db = engine.DocumentDB()
    store.ensure_demo()
    yield


@pytest.fixture
def client(fresh_store):
    from fastapi.testclient import TestClient
    from api.index import app

    with TestClient(app) as c:
        yield c
