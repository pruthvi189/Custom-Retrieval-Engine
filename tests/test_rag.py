"""RAG document endpoints with provider calls faked at the store boundary.

``conftest.fake_embed`` maps a text to a unit vector based on its char sum,
so identical text embeds to the same vector (distance 0) and unrelated text
is orthogonal (distance 1) - the 0.7 threshold is therefore reachable in
both directions without any network access.
"""

DOC_TEXT = "Paris is a city in France. It has the Eiffel Tower."
Q_MATCH = DOC_TEXT
Q_MISS = "zzzz unrelated query"


def insert_doc(client, title="Paris Guide", text=DOC_TEXT):
    r = client.post("/api/doc/insert", json={"title": title, "text": text})
    assert r.status_code == 200, r.text
    return r.json()


class TestDocInsert:
    def test_short_text_single_chunk(self, client):
        body = insert_doc(client)
        assert body == {"ids": [1], "chunks": 1, "dims": 1536}

    def test_missing_fields(self, client):
        for payload in ({}, {"title": "x"}, {"text": "y"}, {"title": "  ", "text": "body"}):
            r = client.post("/api/doc/insert", json=payload)
            assert r.status_code == 400
            assert r.json() == {"error": "need title and text"}

    def test_long_text_multi_chunk(self, client):
        text = " ".join("w%d" % i for i in range(700))
        body = insert_doc(client, text=text)
        assert body["chunks"] >= 2
        assert len(body["ids"]) == body["chunks"]
        assert len(body["ids"]) == len(set(body["ids"]))


class TestDocList:
    def test_empty(self, client):
        assert client.get("/api/doc/list").json() == []

    def test_preview_and_words(self, client):
        insert_doc(client)
        docs = client.get("/api/doc/list").json()
        assert len(docs) == 1
        assert docs[0]["id"] == 1
        assert docs[0]["title"] == "Paris Guide"
        assert docs[0]["words"] == len(DOC_TEXT.split())
        assert docs[0]["preview"] == DOC_TEXT

    def test_preview_truncated_with_ellipsis(self, client):
        insert_doc(client, text=" ".join(["word"] * 200))
        docs = client.get("/api/doc/list").json()
        assert docs[0]["preview"].endswith("\u2026")
        assert len(docs[0]["preview"]) == 121

    def test_multi_chunk_titles_numbered(self, client):
        text = " ".join("w%d" % i for i in range(700))
        insert_doc(client, text=text)
        docs = client.get("/api/doc/list").json()
        assert docs[0]["title"].startswith("Paris Guide [1/")


class TestDocSearch:
    def test_requires_question(self, client):
        r = client.post("/api/doc/search", json={})
        assert r.status_code == 400
        assert r.json() == {"error": "need question"}

    def test_matching_context(self, client):
        insert_doc(client)
        r = client.post("/api/doc/search", json={"question": Q_MATCH})
        body = r.json()
        assert r.status_code == 200
        assert len(body["contexts"]) == 1
        c = body["contexts"][0]
        assert set(c) == {"id", "title", "distance"}
        assert c["id"] == 1
        assert c["title"] == "Paris Guide"
        assert c["distance"] <= 0.7

    def test_threshold_excludes_unrelated(self, client):
        insert_doc(client)
        r = client.post("/api/doc/search", json={"question": Q_MISS})
        assert r.json()["contexts"] == []

    def test_k_param(self, client):
        insert_doc(client)
        r = client.post("/api/doc/search", json={"question": Q_MATCH, "k": 1})
        assert len(r.json()["contexts"]) == 1


class TestDocAsk:
    def test_requires_question(self, client):
        r = client.post("/api/doc/ask", json={})
        assert r.status_code == 400
        assert r.json() == {"error": "need question"}

    def test_empty_database_never_calls_generate(self, client, monkeypatch):
        calls = []
        monkeypatch.setattr("api.providers.generate", lambda p: calls.append(p) or "SHOULD NOT HAPPEN")
        r = client.post("/api/doc/ask", json={"question": "anything"})
        body = r.json()
        assert r.status_code == 200
        assert body["notFound"] is True
        assert body["docCount"] == 0
        assert body["contexts"] == []
        assert "No documents in the database yet" in body["answer"]
        assert calls == []

    def test_grounded_answer_uses_context(self, client, monkeypatch):
        insert_doc(client)
        prompts = []

        def gen(prompt):
            prompts.append(prompt)
            return "The Eiffel Tower is in Paris."

        monkeypatch.setattr("api.providers.generate", gen)
        r = client.post("/api/doc/ask", json={"question": Q_MATCH})
        body = r.json()
        assert r.status_code == 200
        assert body["notFound"] is False
        assert body["docCount"] == 1
        assert len(body["contexts"]) == 1
        assert body["contexts"][0]["id"] == 1
        assert body["answer"] == "The Eiffel Tower is in Paris."
        assert body["model"]
        prompt = prompts[0]
        assert "[1] Paris Guide:" in prompt
        assert DOC_TEXT in prompt
        assert "Question: " + Q_MATCH in prompt

    def test_generation_not_found_flag(self, client, monkeypatch):
        insert_doc(client)
        monkeypatch.setattr("api.providers.generate", lambda p: "Not found in your documents.")
        body = client.post("/api/doc/ask", json={"question": Q_MATCH}).json()
        assert body["notFound"] is True
        assert len(body["contexts"]) == 1

    def test_docs_exist_but_no_close_context(self, client):
        insert_doc(client)
        body = client.post("/api/doc/ask", json={"question": Q_MISS}).json()
        assert body["notFound"] is True
        assert body["contexts"] == []
        assert body["answer"] == "Not found in your documents."
        assert body["docCount"] == 1


class TestDocDelete:
    def test_delete_existing(self, client):
        insert_doc(client)
        assert client.delete("/api/doc/delete/1").json() == {"ok": True}
        assert client.get("/api/doc/list").json() == []

    def test_delete_missing(self, client):
        assert client.delete("/api/doc/delete/999").json() == {"ok": False}

    def test_delete_bad_id(self, client):
        r = client.delete("/api/doc/delete/abc")
        assert r.status_code == 404
        assert r.json() == {"error": "not found"}


class TestWebIngest:
    def test_requires_topic(self, client):
        r = client.post("/api/agent/ingest", json={})
        assert r.status_code == 400
        assert r.json() == {"error": "need topic"}

    def test_no_wiki_results(self, client, monkeypatch):
        monkeypatch.setattr("api.store.wiki_search", lambda topic: [])
        r = client.post("/api/agent/ingest", json={"topic": "definitely not a topic"})
        assert r.status_code == 400
        assert "No Wikipedia results" in r.json()["error"]

    def test_happy_path(self, client, monkeypatch):
        monkeypatch.setattr("api.store.wiki_search", lambda topic: ["Pythagoras"])
        monkeypatch.setattr(
            "api.store.wiki_fetch",
            lambda title: {"title": title, "text": "Pythagoras was a Greek mathematician."},
        )
        r = client.post("/api/agent/ingest", json={"topic": "Pythagoras"})
        body = r.json()
        assert r.status_code == 200, r.text
        assert len(body["added"]) == 1
        a = body["added"][0]
        assert a == {"title": "Pythagoras", "chunks": 1, "stored": 1, "truncated": False}
        assert body["dims"] == 1536
        assert "Stored 1 chunk" in body["message"]
        assert len(client.get("/api/doc/list").json()) == 1

    def test_max_articles(self, client, monkeypatch):
        monkeypatch.setattr("api.store.wiki_search", lambda topic: ["A", "B", "C"])
        monkeypatch.setattr(
            "api.store.wiki_fetch",
            lambda title: {"title": title, "text": title + " is a topic with a single chunk."},
        )
        r = client.post("/api/agent/ingest", json={"topic": "t", "maxArticles": 2})
        body = r.json()
        assert r.status_code == 200
        assert [a["title"] for a in body["added"]] == ["A", "B"]
        assert "Stored 2 chunk(s)" in body["message"]

    def test_graph_point_added_for_article(self, client, monkeypatch):
        monkeypatch.setattr("api.store.wiki_search", lambda topic: ["Pythagoras"])
        monkeypatch.setattr(
            "api.store.wiki_fetch",
            lambda title: {"title": title, "text": "Pythagoras was a Greek mathematician."},
        )
        client.post("/api/agent/ingest", json={"topic": "Pythagoras"})
        metas = [it["metadata"] for it in client.get("/api/items").json()]
        assert "Pythagoras" in metas
