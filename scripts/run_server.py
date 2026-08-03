"""Run the API locally with uvicorn.

Usage:
    python scripts/run_server.py            # http://127.0.0.1:8000
    python scripts/run_server.py --port 3000 --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()
    uvicorn.run("api.index:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
