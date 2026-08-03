"""Brute force index - the baseline every other index gets compared against.

Just stores everything and does a linear scan per query. The API keeps the
pure-Python scan (identical IEEE-754 doubles to the Node original so JSON
contracts stay stable); a numpy-accelerated scan lives in
:mod:`engine.numpy_ops` for the benchmark scripts.
"""

from __future__ import annotations

from .distance import Vector
from .heaps import Entry


class BruteForce:
    def __init__(self) -> None:
        self.items: list = []

    def insert(self, v) -> None:
        self.items.append(v)

    def knn(self, q: Vector, k: int, dist) -> list[Entry]:
        r = [(dist(q, v.embedding), v.id) for v in self.items]
        r.sort(key=lambda t: t[0])
        return [Entry(d=d, id=i) for d, i in r[:k]]

    def remove(self, id: int) -> None:
        self.items = [v for v in self.items if v.id != id]
