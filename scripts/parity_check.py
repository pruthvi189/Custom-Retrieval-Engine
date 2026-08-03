"""Contract parity check against the API.

By default it runs the FastAPI app in-process via TestClient (no server
needed) so it can run in CI. Pass ``--base http://localhost:8000`` to check a
live deployment instead. Verifies the exact behavior the frontend relies on:
routes, JSON shapes, 6-decimal distance rounding, error strings, and the
``parseInt||default`` query quirks. Exits non-zero if anything diverges.

Usage:
    python scripts/parity_check.py
    python scripts/parity_check.py --base http://localhost:8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DIMS = 16
V16 = [0.1] * 16
V16_PARAM = ",".join(str(x) for x in V16)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}  {detail}")


class Client:
    def __init__(self, base: str | None) -> None:
        if base:
            import httpx

            self._c = httpx.Client(base_url=base, timeout=30.0)
        else:
            from fastapi.testclient import TestClient
            from api.index import app

            self._c = TestClient(app)

    def get(self, path, **kw):
        return self._c.get(path, **kw)

    def post(self, path, **kw):
        return self._c.post(path, **kw)

    def delete(self, path, **kw):
        return self._c.delete(path, **kw)

    def options(self, path, **kw):
        return self._c.options(path, **kw)


def run(c: Client) -> None:
    print("[status/stats]")
    s = c.get("/api/status")
    check("status 200", s.status_code == 200)
    body = s.json()
    check("status fields", all(k in body for k in ["groqAvailable", "apiKeySet", "embedAvailable", "embedKeySet", "embedModel", "genModel", "docCount", "docDims", "demoDims", "demoCount"]))
    check("demo dims 16", body["demoDims"] == DIMS)
    stats = c.get("/api/stats").json()
    check("stats count", stats["count"] >= 1)
    check("stats dims", stats["dims"] == DIMS)

    print("[items]")
    items = c.get("/api/items").json()
    check("items list", isinstance(items, list) and len(items) >= 1)
    check("item shape", all(set(i) == {"id", "metadata", "category", "embedding"} for i in items))
    check("item dims", all(len(i["embedding"]) == DIMS for i in items))

    print("[search]")
    r = c.get("/api/search", params={"v": V16_PARAM})
    check("search 200", r.status_code == 200)
    data = r.json()
    check("default k=5", len(data["results"]) == 5)
    check("default algo hnsw", data["algo"] == "hnsw")
    check("default metric cosine", data["metric"] == "cosine")
    check("distance rounded 6dp", all(h["distance"] == round(h["distance"], 6) for h in data["results"]))
    dists = [h["distance"] for h in data["results"]]
    check("results sorted", dists == sorted(dists))
    check("latencyUs >= 0", data["latencyUs"] >= 0)
    check("k=0 falls back to 5", len(c.get("/api/search", params={"v": V16_PARAM, "k": "0"}).json()["results"]) == 5)
    check("k=abc falls back to 5", len(c.get("/api/search", params={"v": V16_PARAM, "k": "abc"}).json()["results"]) == 5)
    bad = c.get("/api/search", params={"v": "1,2,3"})
    check("short vector -> 400", bad.status_code == 400 and bad.json() == {"error": "need 16D vector"})
    echo = c.get("/api/search", params={"v": V16_PARAM, "algo": "kdtree", "metric": "manhattan"}).json()
    check("algo/metric echo", echo["algo"] == "kdtree" and echo["metric"] == "manhattan")

    print("[insert/delete roundtrip]")
    ins = c.post("/api/insert", json={"metadata": "parity check", "category": "test", "embedding": V16})
    check("insert 200", ins.status_code == 200)
    iid = ins.json()["id"]
    check("insert id int", isinstance(iid, int))
    check("items grew", len(c.get("/api/items").json()) == len(items) + 1)
    check("insert empty body -> 400", c.post("/api/insert", json={}).status_code == 400)
    check("insert bad dims -> 400", c.post("/api/insert", json={"metadata": "x", "category": "c", "embedding": [1, 2]}).status_code == 400)
    check("delete ok", c.delete(f"/api/delete/{iid}").json() == {"ok": True})
    check("delete missing ok:false", c.delete("/api/delete/999999").json() == {"ok": False})
    check("delete bad id -> 404", c.delete("/api/delete/abc").json() == {"error": "not found"})

    print("[benchmark + hnsw-info]")
    b = c.get("/api/benchmark", params={"v": V16_PARAM}).json()
    check("benchmark fields", set(b) == {"bruteforceUs", "kdtreeUs", "hnswUs", "itemCount"})
    info = c.get("/api/hnsw-info").json()
    check("hnsw-info nodes", info["nodeCount"] == len(info["nodes"]) and info["nodeCount"] >= 1)
    check("hnsw-info edges", sum(info["edgesPerLayer"]) == len(info["edges"]))

    print("[routing]")
    check("unknown path 404", c.get("/api/nope").status_code == 404)
    check("wrong method 404 (not 405)", c.post("/api/items").status_code == 404)
    check("trailing slash ok", c.get("/api/items").json() == c.get("/api/items/").json())
    opt = c.options("/api/search")
    check("cors preflight", opt.status_code == 204 and opt.headers.get("access-control-allow-origin") == "*")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=None, help="live server base URL; default runs in-process")
    args = p.parse_args()
    run(Client(args.base))
    if FAILURES:
        print(f"\n{len(FAILURES)} parity check(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("\nAll parity checks passed.")


if __name__ == "__main__":
    main()
