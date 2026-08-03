"""numpy-accelerated distance + top-k.

Only used by the benchmark scripts and opt-in fast paths - never by the
server hot path (which stays pure Python to keep float results identical to
the original and cold starts fast). Imported lazily.
"""

from __future__ import annotations


def distance_matrix(rows, q, metric: str = "euclidean"):
    """Compute the distance from ``q`` to every ``rows`` vector.

    ``rows`` is an iterable of items with a list-like ``embedding``.
    Returns ``(matrix, ids)`` where ``matrix[i]`` is the distance to
    ``rows[i]`` and ``ids`` holds the item ids in the same order.
    """
    import numpy as np

    qa = np.asarray(q, dtype=np.float64)
    ids = []
    vecs = []
    for it in rows:
        ids.append(it["id"])
        vecs.append(it["embedding"])
    M = np.asarray(vecs, dtype=np.float64)
    if metric == "cosine":
        qn = np.linalg.norm(qa)
        mn = np.linalg.norm(M, axis=1)
        dots = M @ qa
        with np.errstate(invalid="ignore"):
            sim = dots / (mn * qn)
        sim = np.where((mn == 0) | (qn == 0), 0.0, sim)
        return 1.0 - sim, ids
    if metric == "manhattan":
        return np.abs(M - qa).sum(axis=1), ids
    return np.sqrt(((M - qa) ** 2).sum(axis=1)), ids


def topk(distances, k: int) -> list[int]:
    """Indices of the k smallest distances (stable-ish via argpartition)."""
    import numpy as np

    k = min(k, distances.shape[0])
    if k <= 0:
        return []
    idx = np.argpartition(distances, k - 1)[:k]
    return [int(i) for i in idx[np.argsort(distances[idx])]]
