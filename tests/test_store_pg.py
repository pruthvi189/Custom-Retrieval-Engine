"""Postgres + pgvector layer.

Pure parts (schema constants, ``vec_to_string``) run in every test run; the
integration roundtrip only runs when ``DATABASE_URL`` / ``POSTGRES_URL`` is
set, since it talks to a real database.
"""

import uuid

import pytest

import api.db as db


class TestVecToString:
    def test_bracket_format(self):
        assert db.vec_to_string([1, 2]) == "[1.000000,2.000000]"

    def test_rounds_to_6_decimals(self):
        assert db.vec_to_string([0.123456789, -0.1]) == "[0.123457,-0.100000]"

    def test_floats_from_ints(self):
        assert db.vec_to_string([0]) == "[0.000000]"


class TestDdl:
    def test_items_schema_vector_16(self):
        assert "embedding vector(16)" in db.DDL_ITEMS
        assert "SERIAL PRIMARY KEY" in db.DDL_ITEMS

    def test_docs_schema_vector_1536(self):
        assert "embedding vector(1536)" in db.DDL_DOCS
        assert "SERIAL PRIMARY KEY" in db.DDL_DOCS

    def test_create_if_not_exists(self):
        assert db.DDL_ITEMS.startswith("CREATE TABLE IF NOT EXISTS")
        assert db.DDL_DOCS.startswith("CREATE TABLE IF NOT EXISTS")


class TestDbUrl:
    def test_returns_configured_var(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgres://a:b@h/db")
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        assert db.db_url() == "postgres://a:b@h/db"

    def test_prefers_database_url(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgres://first")
        monkeypatch.setenv("POSTGRES_URL", "postgres://second")
        assert db.db_url() == "postgres://first"

    def test_empty_when_unset(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("POSTGRES_URL", raising=False)
        assert db.db_url() == ""


@pytest.mark.skipif(not db.db_url(), reason="no DATABASE_URL/POSTGRES_URL configured")
class TestPersistenceIntegration:
    @pytest.fixture(autouse=True)
    def real_db(self, monkeypatch):
        monkeypatch.setattr(db, "get_pool", REAL_GET_POOL)

    def test_connection_available(self):
        assert db.get_pool() is not None

    def test_roundtrip(self):
        marker = f"pg-it-{uuid.uuid4().hex}"
        row = db.query(
            "INSERT INTO items (metadata, category, embedding) VALUES (%s, %s, %s::vector) RETURNING id",
            [marker, "test", db.vec_to_string([0.1] * 16)],
        )
        iid = row[0]["id"]
        assert isinstance(iid, int)
        fetched = db.query("SELECT metadata FROM items WHERE id = %s", [iid])
        assert fetched and fetched[0]["metadata"] == marker
        db.query("DELETE FROM items WHERE id = %s", [iid])
        assert db.query("SELECT 1 FROM items WHERE id = %s", [iid]) == []

    def test_doc_chunks_roundtrip(self):
        marker = f"pg-doc-{uuid.uuid4().hex}"
        row = db.query(
            "INSERT INTO doc_chunks (title, text, embedding) VALUES (%s, %s, %s::vector) RETURNING id",
            [marker, "body", db.vec_to_string([0.0] * 1536)],
        )
        iid = row[0]["id"]
        fetched = db.query("SELECT title FROM doc_chunks WHERE id = %s", [iid])
        assert fetched and fetched[0]["title"] == marker
        db.query("DELETE FROM doc_chunks WHERE id = %s", [iid])
