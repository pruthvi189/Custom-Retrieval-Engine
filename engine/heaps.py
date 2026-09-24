"""binary heaps on {d, id} entries, backed by stdlib :mod:`heapq`.

Used to keep the top-k candidates during tree searches without sorting the
whole dataset. Min-heap powers HNSW's candidate/found sets; Max-heap keeps
the k best HNSW/kd-tree results. ``heapq`` owns the ordering: tuples are
``(d, seq, entry)`` for the min-heap and ``(-d, seq, entry)`` for the
max-heap, with ``seq`` breaking ties in insertion order.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    d: float
    id: int


class MinHeap:
    __slots__ = ("a", "_seq")

    def __init__(self) -> None:
        self.a: list[tuple[float, int, Entry]] = []
        self._seq = 0

    def push(self, x: Entry) -> None:
        heapq.heappush(self.a, (x.d, self._seq, x))
        self._seq += 1

    def pop(self) -> Entry | None:
        if not self.a:
            return None
        return heapq.heappop(self.a)[2]

    def peek(self) -> Entry | None:
        return self.a[0][2] if self.a else None

    def __len__(self) -> int:
        return len(self.a)

    @property
    def size(self) -> int:
        return len(self.a)


class MaxHeap:
    __slots__ = ("a", "_seq")

    def __init__(self) -> None:
        self.a: list[tuple[float, int, Entry]] = []
        self._seq = 0

    def push(self, x: Entry) -> None:
        heapq.heappush(self.a, (-x.d, self._seq, x))
        self._seq += 1

    def pop(self) -> Entry | None:
        if not self.a:
            return None
        return heapq.heappop(self.a)[2]

    def peek(self) -> Entry | None:
        return self.a[0][2] if self.a else None

    def __len__(self) -> int:
        return len(self.a)

    @property
    def size(self) -> int:
        return len(self.a)
