"""Smoke test for System 2: POSTs a hand-built trajectory + one keyframe.

Usage (after server is up on a GPU node):
    SMASH_NODE_HOST=<node> python scripts/smoke_system2.py
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
    port = int(os.environ.get("SMASH_S2_PORT", "8002"))
    url = f"http://{host}:{port}/counterfactual"

    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_frame.jpg"
    if not fixture.exists():
        print(f"missing fixture: {fixture}", file=sys.stderr)
        return 2

    b64 = base64.b64encode(fixture.read_bytes()).decode()
    trajectory = [
        {
            "t": 0.0,
            "damage": {"p1": 30.0, "p2": 40.0},
            "stocks": {"p1": 1, "p2": 1},
            "actions": {"p1": {"label": "shield"}, "p2": {"label": "fsmash"}},
            "intent": {"p1": "neutral", "p2": "pressuring"},
        },
        {
            "t": 0.2,
            "damage": {"p1": 45.0, "p2": 40.0},
            "stocks": {"p1": 1, "p2": 1},
            "actions": {"p1": {"label": "neutral"}, "p2": {"label": "neutral"}},
            "intent": {"p1": "neutral", "p2": "neutral"},
        },
        {
            "t": 0.5,
            "damage": {"p1": 45.0, "p2": 40.0},
            "stocks": {"p1": 1, "p2": 1},
            "actions": {"p1": {"label": "roll"}, "p2": {"label": "dash"}},
            "intent": {"p1": "neutral", "p2": "punishing"},
        },
    ]
    payload = {
        "state_trajectory": trajectory,
        "keyframes": [{"image_b64": b64, "t": 0.1}],
        "event_type": "exchange",
    }

    print(f"POST {url}  ({len(trajectory)} states, 1 keyframe)", file=sys.stderr)
    r = httpx.post(url, json=payload, timeout=120.0)
    r.raise_for_status()
    out = r.json()
    print(json.dumps(out, indent=2))

    for k in ("summary", "chosen_action", "counterfactual_action", "frame_data_citations"):
        assert k in out, f"missing top-level key {k}"
    print("smoke_system2 OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
