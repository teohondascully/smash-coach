"""Smoke test for System 1: POSTs tests/fixtures/sample_frame.jpg to /infer.

Usage (after server is up on a GPU node):
    SMASH_NODE_HOST=<node> python scripts/smoke_system1.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import httpx


def main() -> int:
    host = os.environ.get("SMASH_NODE_HOST", "localhost")
    port = int(os.environ.get("SMASH_S1_PORT", "8001"))
    url = f"http://{host}:{port}/infer"

    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_frame.jpg"
    if not fixture.exists():
        print(f"missing fixture: {fixture}", file=sys.stderr)
        return 2

    b64 = base64.b64encode(fixture.read_bytes()).decode()
    payload = {"image_b64": b64, "t": 0.0}

    print(f"POST {url}  ({len(b64)} b64 chars)", file=sys.stderr)
    r = httpx.post(url, json=payload, timeout=60.0)
    r.raise_for_status()
    out = r.json()
    print(json.dumps(out, indent=2))

    # Quick structural sanity check.
    assert "p1" in out and "p2" in out, "missing p1/p2 keys"
    for who in ("p1", "p2"):
        for k in ("action_label", "phase", "confidence", "intent"):
            assert k in out[who], f"missing {who}.{k}"
    print("smoke_system1 OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
