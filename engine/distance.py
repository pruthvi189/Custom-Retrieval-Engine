"""Distance metrics.

Euclidean and manhattan are plain L2/L1; cosine is returned as a "distance"
too (1 - similarity) so the same code path works for all three.
"""

from __future__ import annotations

from math import sqrt
from typing import Callable

Vector = list[float]


def euclidean(a: Vector, b: Vector) -> float:
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return sqrt(s)


def cosine(a: Vector, b: Vector) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    # degenerate vectors: treat as maximally dissimilar
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (sqrt(na) * sqrt(nb))


def manhattan(a: Vector, b: Vector) -> float:
    s = 0.0
    for i in range(len(a)):
        s += abs(a[i] - b[i])
    return s


def get_dist_fn(m: str) -> Callable[[Vector, Vector], float]:
    if m == "cosine":
        return cosine
    if m == "manhattan":
        return manhattan
    return euclidean
