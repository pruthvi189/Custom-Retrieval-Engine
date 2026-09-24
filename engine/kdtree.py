"""kd-tree.

euclidean and manhattan k-NN go through :class:`scipy.spatial.cKDTree` - the
tree build and pruning are library-owned. Cosine has no kd-tree library
equivalent, so that metric keeps the small hand-written axis-cycling tree
below (splits cycle one dimension per level, hyperplane pruning during k-NN).
Deletions rebuild the whole structure from scratch - fine at this scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .distance import Vector, euclidean, manhattan
from .heaps import Entry, MaxHeap

if TYPE_CHECKING:
    from scipy.spatial import cKDTree


class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, v) -> None:
        self.item = v
        self.left = None
        self.right = None


def _minkowski_p(dist) -> int | None:
    """Map a distance function onto scipy's Minkowski ``p`` (None = hand path)."""
    name = getattr(dist, "__name__", "")
    if dist is euclidean or name == "euclidean":
        return 2
    if dist is manhattan or name == "manhattan":
        return 1
    return None


class KDTree:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.root: KDNode | None = None
        self._items: list = []
        self._scipy: cKDTree | None = None

    def _insert(self, n: KDNode | None, v, d: int) -> KDNode:
        if n is None:
            return KDNode(v)
        ax = d % self.dims
        if v.embedding[ax] < n.item.embedding[ax]:
            n.left = self._insert(n.left, v, d + 1)
        else:
            n.right = self._insert(n.right, v, d + 1)
        return n

    def insert_item(self, item) -> None:
        self._items.append(item)
        self.root = self._insert(self.root, item, 0)
        self._scipy = None

    def rebuild(self, items) -> None:
        self._items = list(items)
        self.root = None
        for v in self._items:
            self.root = self._insert(self.root, v, 0)
        self._scipy = None

    def _knn_rec(self, n: KDNode | None, q: Vector, k: int, d: int, dist, heap: MaxHeap) -> None:
        if n is None:
            return
        dn = dist(q, n.item.embedding)
        if heap.size < k or dn < heap.peek().d:
            heap.push(Entry(d=dn, id=n.item.id))
            if heap.size > k:
                heap.pop()
        # descend the side the query falls on, then backtrack if the hyperplane
        # is close enough that a better match could live on the far side
        ax = d % self.dims
        diff = q[ax] - n.item.embedding[ax]
        closer = n.left if diff < 0 else n.right
        farther = n.right if diff < 0 else n.left
        self._knn_rec(closer, q, k, d + 1, dist, heap)
        if heap.size < k or abs(diff) < heap.peek().d:
            self._knn_rec(farther, q, k, d + 1, dist, heap)

    def _knn_hand(self, q: Vector, k: int, dist) -> list[Entry]:
        heap = MaxHeap()
        self._knn_rec(self.root, q, k, 0, dist, heap)
        r = []
        while heap.size:
            r.append(heap.pop())
        r.sort(key=lambda e: e.d)
        return r

    def _knn_scipy(self, q: Vector, k: int, dist, p: int) -> list[Entry]:
        if not self._items:
            return []
        if self._scipy is None:
            from scipy.spatial import cKDTree

            self._scipy = cKDTree(
                np.asarray([v.embedding for v in self._items], dtype=np.float64)
            )
        # over-fetch a few neighbours so the final top-k is ranked with the
        # project's own distance functions (bit-identical to brute force)
        pool = min(len(self._items), k + 8)
        _, idx = self._scipy.query(np.asarray(q, dtype=np.float64), k=pool, p=p)
        out = [
            Entry(d=float(dist(q, self._items[int(i)].embedding)), id=self._items[int(i)].id)
            for i in np.atleast_1d(idx)
        ]
        out.sort(key=lambda e: e.d)
        return out[:k]

    def knn(self, q: Vector, k: int, dist) -> list[Entry]:
        if k <= 0:
            return []
        p = _minkowski_p(dist)
        if p is None:
            return self._knn_hand(q, k, dist)
        return self._knn_scipy(q, k, dist, p)
