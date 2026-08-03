"""Scale + accuracy benchmark for the three search algorithms.

Builds synthetic clustered datasets with numpy, times brute force / kd-tree /
HNSW against growing sizes, reports recall@k (vs brute force), and renders a
matplotlib chart to ``plots/benchmark.png``.

Usage:
    python scripts/benchmark.py --sizes 100 500 1000 2000 4000 --k 5
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from engine import VectorDB, euclidean  # noqa: E402
from engine.numpy_ops import distance_matrix, topk  # noqa: E402


def make_items(n: int, dims: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(8, dims))
    assign = rng.integers(0, 8, size=n)
    vecs = centers[assign] + rng.normal(scale=0.2, size=(n, dims))
    return [
        {"id": i + 1, "embedding": [float(x) for x in vecs[i]], "metadata": f"v{i+1}", "category": "synthetic"}
        for i in range(n)
    ]


def time_once(fn) -> float:
    t = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - t) / 1000.0


def run_size(n: int, dims: int, k: int, seed: int) -> dict:
    random.seed(seed)
    items = make_items(n, dims, seed)
    db = VectorDB(dims)
    for it in items:
        db.insert_raw(it)
    q = items[n // 2]["embedding"]

    truth_ids = {e.id for e in db.bf.knn(q, k, euclidean)}
    hnsw_ids = {e.id for e in db.hnsw.knn(q, k, 50, euclidean)}
    kdt_ids = {e.id for e in db.kdt.knn(q, k, euclidean)}

    return {
        "n": n,
        "bruteforceUs": time_once(lambda: db.bf.knn(q, k, euclidean)),
        "kdtreeUs": time_once(lambda: db.kdt.knn(q, k, euclidean)),
        "hnswUs": time_once(lambda: db.hnsw.knn(q, k, 50, euclidean)),
        "numpyUs": time_once(lambda: topk(distance_matrix(items, q)[0], k)),
        "hnswRecall": len(truth_ids & hnsw_ids) / k,
        "kdtreeRecall": len(truth_ids & kdt_ids) / k,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000, 2000, 4000])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--dims", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="plots")
    args = p.parse_args()

    print(f"{'n':>6} {'bruteforce':>12} {'kdtree':>12} {'hnsw':>12} {'numpy':>12} "
          f"{'hnsw@k':>8} {'kdt@k':>8}")
    results = []
    for n in args.sizes:
        r = run_size(n, args.dims, args.k, args.seed)
        results.append(r)
        print(f"{r['n']:>6} {r['bruteforceUs']:>12.1f} {r['kdtreeUs']:>12.1f} "
              f"{r['hnswUs']:>12.1f} {r['numpyUs']:>12.1f} "
              f"{r['hnswRecall']:>8.2f} {r['kdtreeRecall']:>8.2f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ns = [r["n"] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, label in [("bruteforceUs", "brute force"), ("kdtreeUs", "kd-tree"), ("hnswUs", "hnsw"), ("numpyUs", "numpy brute force")]:
        ax1.plot(ns, [r[key] for r in results], marker="o", label=label)
    ax1.set_xscale("log")
    ax1.set_xlabel("dataset size (n)")
    ax1.set_ylabel("avg query time (us)")
    ax1.set_title("kNN query time vs dataset size")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    ax2.plot(ns, [r["hnswRecall"] for r in results], marker="o", label="hnsw recall@k")
    ax2.plot(ns, [r["kdtreeRecall"] for r in results], marker="s", label="kd-tree recall@k")
    ax2.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("dataset size (n)")
    ax2.set_ylabel("recall@k vs brute force")
    ax2.set_title("Index accuracy")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = outdir / "benchmark.png"
    fig.savefig(out, dpi=120)
    print(f"chart -> {out}")


if __name__ == "__main__":
    main()
