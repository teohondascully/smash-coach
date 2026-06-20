"""Smoke test for the pod-side ops agent.

Assumes the agent is running locally:

    uv run uvicorn server.ops_agent:app --port 9000

Hits all four endpoints and pretty-prints the responses. On a Mac the /gpu
endpoint will return the no-nvidia-smi stub.
"""

from __future__ import annotations

import json
import sys

import httpx

BASE = "http://localhost:9000"
ENDPOINTS = ["/", "/gpu", "/health/s1", "/health/s2", "/disk"]


def main() -> int:
    failures = 0
    with httpx.Client(timeout=5.0) as client:
        for path in ENDPOINTS:
            url = f"{BASE}{path}"
            print(f"=== GET {url} ===")
            try:
                r = client.get(url)
                print(f"status: {r.status_code}")
                try:
                    print(json.dumps(r.json(), indent=2))
                except ValueError:
                    print(r.text)
            except Exception as e:
                print(f"ERROR: {e}")
                failures += 1
            print()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
