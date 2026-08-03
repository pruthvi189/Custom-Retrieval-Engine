import random

import pytest

from engine import (
    DIMS,
    DocumentDB,
    MinHeap,
    MaxHeap,
    VectorDB,
    Entry,
    cosine,
    euclidean,
    load_demo,
    manhattan,
)


def fresh_db() -> VectorDB:
    vdb = VectorDB(DIMS)
    load_demo(vdb)
    return vdb


@pytest.fixture
def vdb():
    return fresh_db()


METRICS = ["euclidean", "cosine", "manhattan"]
ALGOS = ["bruteforce", "kdtree", "hnsw"]


def ids_of(results):
    return [r.id for r in results]


def brute_truth(vdb, q, k, metric):
    dfn = {"euclidean": euclidean, "cosine": cosine, "manhattan": manhattan}[metric]
    return ids_of(vdb.bf.knn(q, k, dfn))


class TestDistance:
    def test_euclidean(self):
        assert euclidean([0, 0], [3, 4]) == 5.0

    def test_cosine_identical_is_zero(self):
        v = [0.1, 0.2, 0.3]
        assert cosine(v, v) == pytest.approx(0.0, abs=1e-12)

    def test_cosine_degenerate_returns_one(self):
        assert cosine([0, 0, 0], [1, 2, 3]) == 1.0
        assert cosine([1, 2, 3], [0, 0, 0]) == 1.0

    def test_manhattan(self):
        assert manhattan([0, 0], [1, 2]) == 3.0

    def test_get_dist_fn_defaults_euclidean(self):
        import engine.distance as d

        assert d.get_dist_fn("nope") is euclidean


class TestHeaps:
    def test_minheap_sorted_pop(self):
        h = MinHeap()
        for d in [3, 1, 4, 1, 5, 9, 2, 6]:
            h.push(Entry(d=d, id=d))
        out = []
        while h.size:
            out.append(h.pop().d)
        assert out == sorted(out)

    def test_maxheap_sorted_pop(self):
        h = MaxHeap()
        for d in [3, 1, 4, 1, 5, 9, 2, 6]:
            h.push(Entry(d=d, id=d))
        out = []
        while h.size:
            out.append(h.pop().d)
        assert out == sorted(out, reverse=True)

    def test_peek(self):
        h = MinHeap()
        h.push(Entry(d=2, id=1))
        h.push(Entry(d=1, id=2))
        assert h.peek().d == 1
        assert h.size == 2

    def test_pop_empty(self):
        assert MinHeap().pop() is None
        assert MaxHeap().pop() is None
        assert MinHeap().peek() is None

    def test_ties_ok(self):
        h = MinHeap()
        for i in range(10):
            h.push(Entry(d=5, id=i))
        assert h.pop().d == 5
        assert h.size == 9


class TestKnnGroundTruth:
    @pytest.mark.parametrize("metric", METRICS)
    def test_kdtree_matches_bruteforce_exactly(self, vdb, metric):
        dfn = {"euclidean": euclidean, "cosine": cosine, "manhattan": manhattan}[metric]
        for item in vdb.all():
            q = item.embedding
            for k in [1, 5, 20]:
                truth = vdb.bf.knn(q, k, dfn)
                got = vdb.kdt.knn(q, k, dfn)
                # exact in the set of ids and the sorted distances; tie order may differ
                assert {e.id for e in truth} == {e.id for e in got}
                assert sorted(e.d for e in truth) == pytest.approx(sorted(e.d for e in got), abs=1e-12)

    @pytest.mark.parametrize("metric", METRICS)
    def test_hnsw_recall_at_5(self, vdb, metric):
        random.seed(0)
        dfn = {"euclidean": euclidean, "cosine": cosine, "manhattan": manhattan}[metric]
        hits = 0
        total = 0
        for item in vdb.all():
            q = item.embedding
            truth = set(brute_truth(vdb, q, 5, metric))
            got = set(ids_of(vdb.hnsw.knn(q, 5, 50, dfn)))
            hits += len(truth & got)
            total += len(truth)
        assert hits / total >= 0.95

    def test_bruteforce_k_gt_n(self, vdb):
        q = vdb.all()[0].embedding
        assert len(vdb.bf.knn(q, 100, euclidean)) == vdb.size()


class TestVectorDB:
    def test_insert_returns_monotonic_ids(self, vdb):
        a = vdb.insert("x", "cs", [0.1] * 16)
        b = vdb.insert("y", "cs", [0.2] * 16)
        assert b == a + 1

    def test_insert_raw_advances_next_id(self):
        vdb = VectorDB(16)
        vdb.insert_raw({"id": 99, "metadata": "m", "category": "c", "embedding": [0.1] * 16})
        assert vdb.insert("n", "c", [0.2] * 16) == 100

    def test_remove_updates_all_indexes(self, vdb):
        target = vdb.all()[0].id
        assert vdb.remove(target) is True
        assert target not in [i.id for i in vdb.all()]
        q = [0.1] * 16
        for algo in ALGOS:
            for r in vdb.search(q, 20, "cosine", algo)["hits"]:
                assert r["id"] != target

    def test_remove_missing_returns_false(self, vdb):
        assert vdb.remove(99999) is False

    def test_kdtree_rebuild_after_delete(self, vdb):
        target = vdb.all()[0].id
        vdb.remove(target)
        q = vdb.all()[0].embedding
        assert ids_of(vdb.kdt.knn(q, 3, euclidean)) == brute_truth(vdb, q, 3, "euclidean")

    def test_search_shape(self, vdb):
        out = vdb.search([0.1] * 16, 5, "cosine", "hnsw")
        assert list(out.keys()) == ["hits", "us", "algo", "metric"]
        assert out["algo"] == "hnsw" and out["metric"] == "cosine"
        assert isinstance(out["us"], float) and out["us"] >= 0

    def test_benchmark_shape(self, vdb):
        out = vdb.benchmark([0.1] * 16, 5, "cosine")
        assert set(out) == {"bruteforceUs", "kdtreeUs", "hnswUs", "itemCount"}
        assert out["itemCount"] == vdb.size()

    def test_hnsw_info_invariants(self, vdb):
        info = vdb.hnsw_info()
        assert info["nodeCount"] == vdb.size()
        assert len(info["nodes"]) == info["nodeCount"]
        assert sum(info["edgesPerLayer"]) == len(info["edges"])
        for e in info["edges"]:
            assert e["src"] < e["dst"]
        for n in info["nodes"]:
            assert "metadata" in n and "category" in n and "maxLyr" in n
        assert info["topLayer"] >= 0


class TestHNSW:
    def test_remove_severs_backrefs(self):
        random.seed(7)
        db = fresh_db()
        hnsw = db.hnsw
        victim = next(iter(hnsw.G.keys()))
        hnsw.remove(victim)
        assert victim not in hnsw.G
        for nd in hnsw.G.values():
            for layer in nd.nbrs:
                assert victim not in layer

    def test_deterministic_with_seed(self):
        random.seed(1)
        a = fresh_db().hnsw.get_info()
        random.seed(1)
        b = fresh_db().hnsw.get_info()
        assert a == b

    def test_reset(self):
        db = fresh_db()
        db.hnsw.reset()
        assert db.hnsw.G == {}
        assert db.hnsw.entry_pt == -1
        assert db.hnsw.top_layer == -1


class TestDocumentDB:
    def test_insert_dims(self):
        docdb = DocumentDB()
        assert docdb.get_dims() == 0
        docdb.insert("t", "body", [0.1] * 1536)
        assert docdb.get_dims() == 1536

    def test_search_respects_threshold(self):
        docdb = DocumentDB()
        near = [1.0, 0.0]
        far = [0.0, 1.0]
        docdb.insert("near", "a", near)
        docdb.insert("far", "b", far)
        hits = docdb.search([1.0, 0.0], 5, 0.5)
        titles = [h["item"].title for h in hits]
        assert "near" in titles and "far" not in titles

    def test_remove(self):
        docdb = DocumentDB()
        docdb.insert("t", "b", [0.1] * 8)
        assert docdb.remove(1) is True
        assert docdb.size() == 0
        assert docdb.remove(1) is False

    def test_empty_search(self):
        assert DocumentDB().search([0.1] * 8, 5) == []

