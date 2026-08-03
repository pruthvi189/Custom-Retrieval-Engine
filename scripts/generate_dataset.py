"""Synthetic dataset generator.

Writes clustered 16D vectors the visualizer / API can ingest, using numpy
for the heavy lifting. Outputs ``data/dataset.json`` (insert-ready objects)
and ``data/dataset.csv`` (id, category, 16 floats).

Usage:
    python scripts/generate_dataset.py --n 500 --clusters 8 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import DIMS  # noqa: E402


def round6(x: float) -> float:
    return round(float(x), 6)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=500, help="number of vectors")
    p.add_argument("--clusters", type=int, default=8, help="gaussian cluster centers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dims", type=int, default=DIMS)
    p.add_argument("--outdir", type=str, default="data")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    centers = rng.normal(loc=0.0, scale=1.0, size=(args.clusters, args.dims))
    assign = rng.integers(0, args.clusters, size=args.n)
    vectors = centers[assign] + rng.normal(scale=0.2, size=(args.n, args.dims))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.n):
        emb = [round6(v) for v in vectors[i]]
        rows.append(
            {
                "id": i + 1,
                "metadata": f"synthetic {i + 1}",
                "category": f"cluster-{int(assign[i])}",
                "embedding": emb,
            }
        )

    with (outdir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))
    with (outdir / "dataset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category"] + [f"d{i}" for i in range(args.dims)])
        for r in rows:
            w.writerow([r["id"], r["category"], *r["embedding"]])

    print(f"wrote {args.n} vectors ({args.dims}D, {args.clusters} clusters) to {outdir}/")


if __name__ == "__main__":
    main()
