"""Tests for mac.state."""
from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

from mac.state import (
    ActionState,
    Intent,
    Phase,
    PlayerState,
    StateBuffer,
    StateT,
)


def make_state(t: float) -> StateT:
    return StateT(
        t=t,
        damage={"p1": 0.0, "p2": 0.0},
        stocks={"p1": 1, "p2": 1},
        positions={
            "p1": PlayerState(x=0.0, y=0.0, facing="right", airborne=False),
            "p2": PlayerState(x=100.0, y=0.0, facing="left", airborne=False),
        },
        actions={
            "p1": ActionState(label="neutral"),
            "p2": ActionState(label="neutral"),
        },
    )


def test_state_t_round_trip():
    s = StateT(
        t=1.23,
        damage={"p1": 45.0, "p2": 12.5},
        stocks={"p1": 1, "p2": 1},
        positions={
            "p1": PlayerState(
                x=100.0, y=200.0, facing="right", airborne=False, vx=0.0, vy=0.0
            ),
            "p2": PlayerState(
                x=300.0, y=200.0, facing="left", airborne=True, vx=-1.5, vy=2.0
            ),
        },
        actions={
            "p1": ActionState(
                label="neutral", phase="neutral", confidence=0.9, onset_estimate_t=1.0
            ),
            "p2": ActionState(
                label="fsmash", phase="startup", confidence=0.8, onset_estimate_t=1.1
            ),
        },
        intent={"p1": "neutral", "p2": "pressuring"},
    )
    blob = s.model_dump_json()
    s2 = StateT.model_validate_json(blob)
    assert s2.actions["p2"].label == "fsmash"
    # derived was populated by the model_validator
    assert s2.derived.distance > 0


def test_compute_derived_distance_and_facing():
    s = make_state(0.0)
    # p1 at x=0 facing right, p2 at x=100 -> facing
    assert s.derived.distance == 100.0
    assert s.derived.relative_facing == "facing"

    s2 = StateT(
        t=0.0,
        damage={"p1": 0, "p2": 0},
        stocks={"p1": 1, "p2": 1},
        positions={
            "p1": PlayerState(x=200.0, y=0, facing="right", airborne=False),
            "p2": PlayerState(x=100.0, y=0, facing="left", airborne=False),
        },
        actions={
            "p1": ActionState(label="neutral"),
            "p2": ActionState(label="neutral"),
        },
    )
    # p1 right-facing but p2 is to its left -> back-turned
    assert s2.derived.relative_facing == "back-turned"
    assert s2.derived.distance == 100.0


def test_state_buffer_eviction():
    buf = StateBuffer(window_seconds=1.0)
    for i in range(10):
        buf.push(make_state(t=i * 0.2))
    # latest t = 1.8, cutoff = 0.8 -> only t >= 0.8 kept
    times = [s.t for s in buf._buf]
    assert all(t >= 0.8 - 1e-9 for t in times)
    assert times[-1] == 10 * 0.2 - 0.2  # 1.8


def test_state_buffer_window():
    buf = StateBuffer(window_seconds=10.0)
    for i in range(20):
        buf.push(make_state(t=i * 0.2))
    w = buf.window(1.0, 2.0)
    assert all(1.0 <= s.t <= 2.0 for s in w)
    assert len(w) >= 1


def test_state_buffer_latest():
    buf = StateBuffer(window_seconds=10.0)
    assert buf.latest() is None
    buf.push(make_state(t=0.0))
    buf.push(make_state(t=0.5))
    assert buf.latest().t == 0.5


def test_phase_intent_match_action_vocab():
    vocab = json.loads(
        Path(__file__).resolve().parent.parent.joinpath("data/action_vocab.json").read_text()
    )
    assert set(get_args(Phase)) == set(vocab["phases"])
    assert set(get_args(Intent)) == set(vocab["intents"])
