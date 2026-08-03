"""VectorDB / DocumentDB - the two store-level indexes.

VectorDB is the 16D demo index. It keeps all three algorithms in sync so any
endpoint can compare them side by side. Items live in one source of truth
(``store``), and each index gets a copy of it.

DocumentDB holds chunks of real text embedded with OpenRouter (1536D). Only
brute-force search for now; doc sets here are small enough that it's fine.
"""

from __future__ import annotations

import time

from .distance import Vector, cosine, get_dist_fn
from .bruteforce import BruteForce
from .kdtree import KDTree
from .hnsw import HNSW
from .item import Item


def _as_item(item) -> Item:
    if isinstance(item, Item):
        return item
    return Item(
        id=item["id"],
        embedding=item.get("embedding", []),
        metadata=item.get("metadata", ""),
        category=item.get("category", ""),
        title=item.get("title"),
        text=item.get("text"),
    )


class VectorDB:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.store: dict[int, Item] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.next_id = 1

    def insert(self, meta: str, cat: str, emb: Vector) -> int:
        id = self.next_id
        self.next_id += 1
        self.insert_raw(Item(id=id, metadata=meta, category=cat, embedding=emb))
        return id

    def insert_raw(self, item) -> None:
        item = _as_item(item)
        self.store[item.id] = item
        self.bf.insert(item)
        self.kdt.insert_item(item)
        self.hnsw.insert(item, cosine)
        if item.id >= self.next_id:
            self.next_id = item.id + 1

    def remove(self, id: int) -> bool:
        if id not in self.store:
            return False
        del self.store[id]
        self.bf.remove(id)
        self.hnsw.remove(id)
        self.kdt.rebuild(list(self.store.values()))
        return True

    def reset(self) -> None:
        self.store = {}
        self.bf = BruteForce()
        self.kdt = KDTree(self.dims)
        self.hnsw = HNSW(16, 200)
        self.next_id = 1

    def search(self, q: Vector, k: int, metric: str, algo: str) -> dict:
        dfn = get_dist_fn(metric)
        t0 = time.perf_counter_ns()
        if algo == "bruteforce":
            raw = self.bf.knn(q, k, dfn)
        elif algo == "kdtree":
            raw = self.kdt.knn(q, k, dfn)
        else:
            raw = self.hnsw.knn(q, k, 50, dfn)
        us = (time.perf_counter_ns() - t0) / 1000.0
        hits = []
        for entry in raw:
            it = self.store.get(entry.id)
            if it is not None:
                hits.append({
                    "id": it.id,
                    "metadata": it.metadata,
                    "category": it.category,
                    "embedding": it.embedding,
                    "distance": entry.d,
                })
        return {"hits": hits, "us": us, "algo": algo, "metric": metric}

    def benchmark(self, q: Vector, k: int, metric: str) -> dict:
        dfn = get_dist_fn(metric)

        def time_fn(fn) -> float:
            t = time.perf_counter_ns()
            fn()
            return (time.perf_counter_ns() - t) / 1000.0

        return {
            "bruteforceUs": time_fn(lambda: self.bf.knn(q, k, dfn)),
            "kdtreeUs": time_fn(lambda: self.kdt.knn(q, k, dfn)),
            "hnswUs": time_fn(lambda: self.hnsw.knn(q, k, 50, dfn)),
            "itemCount": len(self.store),
        }

    def all(self) -> list[Item]:
        return list(self.store.values())

    def hnsw_info(self) -> dict:
        return self.hnsw.get_info()

    def size(self) -> int:
        return len(self.store)


class DocumentDB:
    def __init__(self) -> None:
        self.store: dict[int, Item] = {}
        self.bf = BruteForce()
        self.next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: Vector) -> int:
        if not self.dims:
            self.dims = len(emb)
        id = self.next_id
        self.next_id += 1
        item = Item(id=id, title=title, text=text, embedding=emb)
        self.store[id] = item
        self.bf.insert(item)
        return id

    def remove(self, id: int) -> bool:
        if id not in self.store:
            return False
        del self.store[id]
        self.bf.remove(id)
        return True

    def search(self, q: Vector, k: int, max_dist: float = 0.7) -> list[dict]:
        if not self.store:
            return []
        raw = self.bf.knn(q, k, cosine)
        out = []
        for entry in raw:
            it = self.store.get(entry.id)
            if it is not None and entry.d <= max_dist:
                out.append({"distance": entry.d, "item": it})
        return out

    def all(self) -> list[Item]:
        return list(self.store.values())

    def size(self) -> int:
        return len(self.store)

    def get_dims(self) -> int:
        return self.dims
