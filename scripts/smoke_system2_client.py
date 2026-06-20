"""Programmatic client-style smoke for System 2.

Same wire shape as scripts/smoke_system2.py, but exposes a `call()` function so
the Mac orchestrator can use it as a reference for the live System2 client.

Usage (after server is up):
    SMASH_NODE_HOST=<node> python scripts/smoke_system2_client.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx


def call(
    host: str | None = None,
    port: int | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    host = host or os.environ.get("SMASH_NODE_HOST", "localhost")
    port = port or int(os.environ.get("SMASH_S2_PORT", "8002"))
    url = f"http://{host}:{port}/counterfactual"

    fixture = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "sample_frame.jpg"
    )
    b64 = base64.b64encode(fixture.read_bytes()).decode()

    trajectory = [
        {
            "t": 0.0,
            "damage": {"p1": 60.0, "p2": 80.0},
            "stocks": {"p1": 1, "p2": 1},
            "actions": {"p1": {"label": "neutral"}, "p2": {"label": "fair"}},
            "intent": {"p1": "neutral", "p2": "pressuring"},
        },
        {
            "t": 0.3,
            "damage": {"p1": 75.0, "p2": 80.0},
            "stocks": {"p1": 1, "p2": 1},
            "actions": {"p1": {"label": "airdodge"}, "p2": {"label": "neutral"}},
            "intent": {"p1": "recovering", "p2": "ledge-trapping"},
        },
        {
            "t": 0.8,
            "damage": {"p1": 75.0, "p2": 80.0},
            "stocks": {"p1": 0, "p2": 1},
            "actions": {"p1": {"label": "unknown"}, "p2": {"label": "bair"}},
            "intent": {"p1": "recovering", "p2": "punishing"},
        },
    ]

    payload = {
        "state_trajectory": trajectory,
        "keyframes": [{"image_b64": b64, "t": 0.4}],
        "event_type": "stock_loss",
    }

    r = httpx.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main() -> int:
    out = call()
    print(json.dumps(out, indent=2))
    for k in (
        "summary",
        "chosen_action",
        "counterfactual_action",
        "frame_data_citations",
    ):
        assert k in out, f"missing top-level key {k}"
    print("smoke_system2_client OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
