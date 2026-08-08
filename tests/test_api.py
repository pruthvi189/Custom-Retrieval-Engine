"""API contract tests against the FastAPI port of the original handler.js.

These pin the exact HTTP behavior the frontend depends on: same routes, same
query-param parsing quirks (``parseInt||default``, ``Number.isFinite``),
compact JSON, 6-decimal distances, and the "everything is 404" routing rules.
"""

V16 = [0.1] * 16
V16B = [float(i) / 16 for i in range(16)]

V16_PARAM = ",".join(str(x) for x in V16)


class TestParityShims:
    def test_round6(self):
        import api.index as ix

        assert ix.round6(0.12345678) == 0.123457
        assert ix.round6(1.5) == 1.5
        assert ix.round6(-0.1234567) == -0.123457

    def test_parse_v(self):
        import api.index as ix

        assert ix.parse_v(None) == [0.0]
        assert ix.parse_v("") == [0.0]
        assert ix.parse_v("1,2,3") == [1.0, 2.0, 3.0]
        assert ix.parse_v("1,abc,3") == [1.0, 3.0]
        assert ix.parse_v("1,,2") == [1.0, 0.0, 2.0]
        assert ix.parse_v(" 1 , 2 , 3 ") == [1.0, 2.0, 3.0]
        assert ix.parse_v("nan,inf") == []
        assert ix.parse_v("1e2, -2.5") == [100.0, -2.5]

    def test_parse_int(self):
        import api.index as ix

        assert ix.parse_int(None, 5) == 5
        assert ix.parse_int("", 5) == 5
        assert ix.parse_int("0", 5) == 5
        assert ix.parse_int("abc", 5) == 5
        assert ix.parse_int("  7", 5) == 7
        assert ix.parse_int("3", 5) == 3
        assert ix.parse_int("42abc", 5) == 42


class TestItems:
    def test_demo_items(self, client):
        items = client.get("/api/items").json()
        assert len(items) == 20
        for it in items:
            assert set(it) == {"id", "metadata", "category", "embedding"}
            assert len(it["embedding"]) == 16


class TestSearch:
    def test_requires_16d(self, client):
        r = client.get("/api/search", params={"v": "1,2,3"})
        assert r.status_code == 400
        assert r.json() == {"error": "need 16D vector"}

    def test_missing_v_is_400(self, client):
        r = client.get("/api/search", params={})
        assert r.status_code == 400
        assert r.json() == {"error": "need 16D vector"}

    def test_default_params(self, client):
        r = client.get("/api/search", params={"v": V16_PARAM})
        body = r.json()
        assert r.status_code == 200
        assert body["algo"] == "hnsw"
        assert body["metric"] == "cosine"
        assert len(body["results"]) == 5
        assert body["latencyUs"] >= 0

    def test_k_parsing(self, client):
        params = {"v": V16_PARAM}
        assert len(client.get("/api/search", params={**params, "k": "0"}).json()["results"]) == 5
        assert len(client.get("/api/search", params={**params, "k": "abc"}).json()["results"]) == 5
        assert len(client.get("/api/search", params={**params, "k": "2"}).json()["results"]) == 2
        assert len(client.get("/api/search", params={**params, "k": "50"}).json()["results"]) == 20

    def test_result_shape_and_rounding(self, client):
        r = client.get(
            "/api/search",
            params={"v": ",".join(str(x) for x in V16B), "k": "5", "metric": "euclidean", "algo": "bruteforce"},
        )
        body = r.json()
        assert body["algo"] == "bruteforce"
        assert body["metric"] == "euclidean"
        results = body["results"]
        assert len(results) == 5
        for h in results:
            assert set(h) == {"id", "metadata", "category", "distance", "embedding"}
            assert h["distance"] == round(h["distance"], 6)
            assert len(h["embedding"]) == 16
        dists = [h["distance"] for h in results]
        assert dists == sorted(dists)

    def test_algo_and_metric_echo(self, client):
        r = client.get(
            "/api/search",
            params={"v": V16_PARAM, "algo": "kdtree", "metric": "manhattan"},
        )
        body = r.json()
        assert body["algo"] == "kdtree"
        assert body["metric"] == "manhattan"
        assert len(body["results"]) == 5

    def test_hnsw_returns_correct_count(self, client):
        r = client.get("/api/search", params={"v": V16_PARAM, "algo": "hnsw"})
        assert len(r.json()["results"]) == 5


class TestInsert:
    def test_insert_ok(self, client):
        r = client.post("/api/insert", json={"metadata": "hello world", "category": "cs", "embedding": V16B})
        assert r.status_code == 200
        iid = r.json()["id"]
        assert isinstance(iid, int)
        items = client.get("/api/items").json()
        assert len(items) == 21
        assert items[-1]["metadata"] == "hello world"

    def test_invalid_body(self, client):
        assert client.post("/api/insert", json={}).status_code == 400
        assert client.post("/api/insert", json={"metadata": "x", "category": "c", "embedding": [1, 2]}).status_code == 400
        assert client.post("/api/insert", json={"metadata": "  ", "category": "c", "embedding": V16}).status_code == 400
        assert client.post("/api/insert", json={"metadata": "x", "category": "c", "embedding": "notalist"}).status_code == 400

    def test_invalid_json_body(self, client):
        r = client.post("/api/insert", content="{not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 400
        assert r.json() == {"error": "invalid body"}

    def test_non_object_body(self, client):
        r = client.post("/api/insert", json=[1, 2, 3])
        assert r.status_code == 400


class TestDelete:
    def test_delete_ok(self, client):
        iid = client.post("/api/insert", json={"metadata": "temp", "category": "x", "embedding": V16}).json()["id"]
        assert client.delete(f"/api/delete/{iid}").json() == {"ok": True}
        assert len(client.get("/api/items").json()) == 20

    def test_delete_missing(self, client):
        assert client.delete("/api/delete/99999").json() == {"ok": False}

    def test_delete_bad_id(self, client):
        r = client.delete("/api/delete/notanumber")
        assert r.status_code == 404
        assert r.json() == {"error": "not found"}


class TestBenchmark:
    def test_benchmark_ok(self, client):
        r = client.get("/api/benchmark", params={"v": V16_PARAM})
        body = r.json()
        assert r.status_code == 200
        assert set(body) == {"bruteforceUs", "kdtreeUs", "hnswUs", "itemCount"}
        assert body["itemCount"] == 20
        assert all(v >= 0 for k, v in body.items() if k != "itemCount")

    def test_benchmark_bad_vector(self, client):
        r = client.get("/api/benchmark", params={"v": "1,2,3"})
        assert r.status_code == 400
        assert r.json() == {"error": "need 16D vector"}


class TestHnswInfo:
    def test_shape(self, client):
        body = client.get("/api/hnsw-info").json()
        assert body["nodeCount"] == 20
        assert len(body["nodes"]) == 20
        assert body["topLayer"] >= 0
        assert sum(body["edgesPerLayer"]) == len(body["edges"])
        for e in body["edges"]:
            assert e["src"] < e["dst"]


class TestStatusStats:
    def test_status(self, client):
        body = client.get("/api/status").json()
        for key in [
            "groqAvailable",
            "apiKeySet",
            "embedAvailable",
            "embedKeySet",
            "embedModel",
            "genModel",
            "docCount",
            "docDims",
            "demoDims",
            "demoCount",
        ]:
            assert key in body
        assert body["demoDims"] == 16
        assert body["demoCount"] == 20
        assert body["docCount"] == 0
        assert body["embedModel"] == "openai/text-embedding-3-small"
        assert body["genModel"] == "meta-llama/llama-3.3-70b-instruct"

    def test_stats(self, client):
        body = client.get("/api/stats").json()
        assert body == {
            "count": 20,
            "dims": 16,
            "algorithms": ["bruteforce", "kdtree", "hnsw"],
            "metrics": ["euclidean", "cosine", "manhattan"],
        }


class TestRouting:
    def test_unknown_path_404(self, client):
        r = client.get("/api/nope")
        assert r.status_code == 404
        assert r.json() == {"error": "not found"}

    def test_wrong_method_is_404_not_405(self, client):
        r = client.post("/api/items")
        assert r.status_code == 404
        assert r.json() == {"error": "not found"}
        assert client.put("/api/search").status_code == 404
        assert client.delete("/api/items").status_code == 404

    def test_trailing_slash_normalized(self, client):
        assert client.get("/api/items").json() == client.get("/api/items/").json()

    def test_cors_preflight(self, client):
        r = client.options("/api/search")
        assert r.status_code == 204
        assert r.headers["access-control-allow-origin"] == "*"
        assert r.headers["access-control-allow-methods"] == "GET, POST, DELETE, OPTIONS"
