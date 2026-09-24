"""Distance metrics, vectorized with numpy.

Euclidean and manhattan are plain L2/L1; cosine is returned as a "distance"
too (1 - similarity) so the same code path works for all three. Results are
returned as builtin ``float`` (numpy scalars are not JSON-serializable).
"""

from __future__ import annotations

from math import sqrt
from typing import Callable

import numpy as np

Vector = list[float]


def euclidean(a: Vector, b: Vector) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(sqrt(float(diff @ diff)))


def cosine(a: Vector, b: Vector) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    dot = float(va @ vb)
    na = float(va @ va)
    nb = float(vb @ vb)
    # degenerate vectors: treat as maximally dissimilar
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (sqrt(na) * sqrt(nb))


def manhattan(a: Vector, b: Vector) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.abs(diff).sum())


def get_dist_fn(m: str) -> Callable[[Vector, Vector], float]:
    if m == "cosine":
        return cosine
    if m == "manhattan":
        return manhattan
    return euclidean
