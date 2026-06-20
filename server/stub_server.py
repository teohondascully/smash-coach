"""Local stub of System 1 + System 2 endpoints.

Lets us end-to-end test mac.main on a laptop without the GPU pod.
Returns schema-conformant JSON with plausible values that change over
time so triggers fire and the rewind card renders.

Run:
    uv run uvicorn server.stub_server:app --host 0.0.0.0 --port 8001 &
    uv run uvicorn server.stub_server:app --host 0.0.0.0 --port 8002 &
    # or simpler:
    ./scripts/run_stub.sh
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from server.prompts.system1 import COARSE_ACTIONS
from server.prompts.system2 import COUNTERFACTUAL_SCHEMA


app = FastAPI(title="Smash Coach STUB (S1+S2 on the same port)")


# ---------------------------------------------------------------------------
# Shared "world state" — drifts over time so the HUD and triggers do
# something interesting during a dry run.
# ---------------------------------------------------------------------------


class _World:
    def __init__(self) -> None:
        self.start = time.monotonic()
        # Damage drifts upward; resets after stock loss.
        self.damage = {"p1": 0, "p2": 0}
        self.stocks = {"p1": 1, "p2": 1}
        self.last_ev_t = 0.0
        self._last_damage_bump = self.start

    def step(self) -> None:
        """Advance state on each inference call."""
        now = time.monotonic()
        # Damage climbs every ~2s by 5-15%, occasionally bigger jumps.
        if now - self._last_damage_bump > 2.0:
            self._last_damage_bump = now
            who = random.choice(("p1", "p2"))
            if random.random() < 0.1:
                # Big hit: triggers an "exchange" event.
                self.damage[who] = min(999, self.damage[who] + random.randint(35, 60))
            else:
                self.damage[who] += random.randint(3, 15)


_world = _World()


# ---------------------------------------------------------------------------
# System 1: /infer
# ---------------------------------------------------------------------------


class FrameIn(BaseModel):
    images_b64: list[str]
    ts: list[float]


@app.post("/infer")
def infer(req: FrameIn) -> dict:
    """Return a schema-conformant fake S1 response. ~30ms artificial delay
    so the dispatcher's rate limiting and async path are exercised."""
    _world.step()

    # Coarse states matching System 1's real (post-rework) output contract.
    # Mostly "idle"; occasionally an active state so the HUD has something to
    # render and triggers can fire.
    def _pick() -> str:
        if random.random() < 0.4:
            return random.choice([a for a in COARSE_ACTIONS if a not in ("idle", "unknown")])
        return "idle"

    p1_label = _pick()
    p2_label = _pick()
    phases = ["startup", "active", "endlag", "neutral"]

    time.sleep(0.03)  # simulate VLM latency

    return {
        "p1": {
            "damage_pct": int(_world.damage["p1"]),
            "action_label": p1_label,
            "phase": "neutral" if p1_label == "idle" else random.choice(phases),
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "intent": "neutral",
        },
        "p2": {
            "damage_pct": int(_world.damage["p2"]),
            "action_label": p2_label,
            "phase": "neutral" if p2_label == "idle" else random.choice(phases),
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "intent": "neutral",
        },
    }


# ---------------------------------------------------------------------------
# System 2: /counterfactual
# ---------------------------------------------------------------------------


class Keyframe(BaseModel):
    image_b64: str
    t: float


class CounterfactualReq(BaseModel):
    state_trajectory: list[dict]
    keyframes: list[Keyframe]
    event_type: str


_FRAME_DATA_CACHE: dict[str, Any] | None = None


def _frame_data() -> dict[str, Any]:
    global _FRAME_DATA_CACHE
    if _FRAME_DATA_CACHE is not None:
        return _FRAME_DATA_CACHE
    path = Path(__file__).resolve().parents[1] / "data" / "frame_data.json"
    _FRAME_DATA_CACHE = json.loads(path.read_text()) if path.exists() else {}
    return _FRAME_DATA_CACHE


def _pick_citation(char: str) -> dict[str, str] | None:
    moves = _frame_data().get(char, [])
    if not moves:
        return None
    m = random.choice(moves)
    return {
        "character": char,
        "move": m["name"],
        "stat": "startup_f",
        "value": str(m.get("startup_f", "?")),
    }


@app.post("/counterfactual")
def counterfactual(req: CounterfactualReq) -> dict:
    """Schema-conformant fake counterfactual. ~2s delay to mirror the real
    72B AWQ call so the rewind card's 'thinking...' state is exercised."""
    time.sleep(2.0)

    chosen_player = "p1" if req.event_type == "stock_loss" else random.choice(("p1", "p2"))
    chosen_char = "toon_link" if chosen_player == "p1" else "ike"
    attacker_char = "ike" if chosen_char == "toon_link" else "toon_link"

    chosen_move = random.choice([m["name"] for m in _frame_data().get(attacker_char, [{"name": "fsmash"}])])
    cf_move = random.choice(("spotdodge", "shield", "roll"))

    cits = [c for c in (_pick_citation(attacker_char), _pick_citation(chosen_char)) if c]

    return {
        "summary": (
            f"You ate {attacker_char}'s {chosen_move} during a {req.event_type}. "
            f"A {cf_move} would have given you tempo back."
        )[:400],
        "chosen_action": {
            "player": chosen_player,
            "action_label": chosen_move,
            "frame_t": req.state_trajectory[-1].get("t", 0.0) if req.state_trajectory else 0.0,
            "reasoning": "Predictable approach + late shield drop.",
        },
        "counterfactual_action": {
            "action_label": cf_move,
            "rationale": f"Beats {chosen_move} on reaction at this distance.",
            "qualitative_likelihood": random.choice(["likely", "plausible"]),
        },
        "frame_data_citations": cits[:4],
    }


# ---------------------------------------------------------------------------
# Smoke endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> dict:
    return {"stub": "smash-coach", "endpoints": ["/infer", "/counterfactual"]}


@app.get("/health")
def health() -> dict:
    return {"ok": True}
