"""kd-tree.

Splits on one dimension per level (axis cycles through 0..dims), so lookups
are O(log n) in the best case. Deletions are handled by rebuilding the whole
tree from scratch - fine at this scale.
"""

from __future__ import annotations

from .distance import Vector
from .heaps import Entry, MaxHeap


class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, v) -> None:
        self.item = v
        self.left = None
        self.right = None


class KDTree:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.root: KDNode | None = None

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
        self.root = self._insert(self.root, item, 0)

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

    def knn(self, q: Vector, k: int, dist) -> list[Entry]:
        heap = MaxHeap()
        self._knn_rec(self.root, q, k, 0, dist, heap)
        r = []
        while heap.size:
            r.append(heap.pop())
        r.sort(key=lambda e: e.d)
        return r

    def rebuild(self, items) -> None:
        self.root = None
        for v in items:
            self.root = self._insert(self.root, v, 0)
