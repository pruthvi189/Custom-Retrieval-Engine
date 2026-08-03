"""Hand-rolled binary heaps on {d, id} entries.

Used to keep the top-k candidates during tree searches without sorting the
whole dataset. Min-heap powers HNSW's candidate/found sets; Max-heap keeps
the k best HNSW/kd-tree results. Written from scratch (no ``heapq``) to
match the project's "implemented by hand" identity and behavior.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    d: float
    id: int


class MinHeap:
    __slots__ = ("a",)

    def __init__(self) -> None:
        self.a: list[Entry] = []

    def push(self, x: Entry) -> None:
        a = self.a
        a.append(x)
        i = len(a) - 1
        while i > 0:
            p = (i - 1) >> 1
            if a[i].d >= a[p].d:
                break
            a[i], a[p] = a[p], a[i]
            i = p

    def pop(self) -> Entry | None:
        a = self.a
        if not a:
            return None
        top = a[0]
        last = a.pop()
        if a:
            a[0] = last
            i = 0
            n = len(a)
            while True:
                l = 2 * i + 1
                r = l + 1
                m = i
                if l < n and a[l].d < a[m].d:
                    m = l
                if r < n and a[r].d < a[m].d:
                    m = r
                if m == i:
                    break
                a[i], a[m] = a[m], a[i]
                i = m
        return top

    def peek(self) -> Entry | None:
        return self.a[0] if self.a else None

    def __len__(self) -> int:
        return len(self.a)

    @property
    def size(self) -> int:
        return len(self.a)


class MaxHeap:
    __slots__ = ("a",)

    def __init__(self) -> None:
        self.a: list[Entry] = []

    def push(self, x: Entry) -> None:
        a = self.a
        a.append(x)
        i = len(a) - 1
        while i > 0:
            p = (i - 1) >> 1
            if a[i].d <= a[p].d:
                break
            a[i], a[p] = a[p], a[i]
            i = p

    def pop(self) -> Entry | None:
        a = self.a
        if not a:
            return None
        top = a[0]
        last = a.pop()
        if a:
            a[0] = last
            i = 0
            n = len(a)
            while True:
                l = 2 * i + 1
                r = l + 1
                m = i
                if l < n and a[l].d > a[m].d:
                    m = l
                if r < n and a[r].d > a[m].d:
                    m = r
                if m == i:
                    break
                a[i], a[m] = a[m], a[i]
                i = m
        return top

    def peek(self) -> Entry | None:
        return self.a[0] if self.a else None

    def __len__(self) -> int:
        return len(self.a)

    @property
    def size(self) -> int:
        return len(self.a)
