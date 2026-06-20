"""State schema for the live smash coach.

Defines $s_t$ pydantic v2 models and a rolling time-windowed buffer.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Facing = Literal["left", "right"]

# NOTE: Literal values below MUST match data/action_vocab.json phases & intents arrays.
# The test test_phase_intent_match_action_vocab guards against drift.
Phase = Literal["startup", "active", "endlag", "neutral", "unknown"]
Intent = Literal[
    "pressuring",
    "ledge-trapping",
    "neutral",
    "recovering",
    "punishing",
    "unknown",
]


class PlayerState(BaseModel):
    x: float
    y: float
    facing: Facing
    airborne: bool
    vx: float = 0.0
    vy: float = 0.0


class ActionState(BaseModel):
    label: str  # closed vocab enforced by S1 prompt grammar
    phase: Phase = "unknown"
    confidence: float = 0.0
    onset_estimate_t: float = 0.0


class Derived(BaseModel):
    distance: float = 0.0
    relative_facing: Literal["facing", "back-turned", "mixed"] = "mixed"
    ledge_owner: Optional[Literal["p1", "p2"]] = None
    stage_control_estimate: float = 0.5  # 0 = p2 dominates, 1 = p1 dominates
    active_punish_window_for: Optional[Literal["p1", "p2"]] = None


class StateT(BaseModel):
    t: float
    damage: dict[str, float]
    stocks: dict[str, int]
    positions: dict[str, PlayerState]
    actions: dict[str, ActionState]
    intent: dict[str, Intent] = Field(
        default_factory=lambda: {"p1": "neutral", "p2": "neutral"}
    )
    derived: Derived = Field(default_factory=Derived)
    controller_input_t: Optional[dict] = None  # reserved for future

    @model_validator(mode="after")
    def compute_derived(self) -> "StateT":
        p1 = self.positions.get("p1")
        p2 = self.positions.get("p2")
        if p1 is None or p2 is None:
            return self
        self.derived.distance = math.hypot(p1.x - p2.x, p1.y - p2.y)
        if (p1.facing == "right" and p1.x < p2.x) or (
            p1.facing == "left" and p1.x > p2.x
        ):
            self.derived.relative_facing = "facing"
        else:
            self.derived.relative_facing = "back-turned"
        return self


class StateBuffer:
    """Time-windowed rolling buffer of StateT, evicting by timestamp."""

    def __init__(self, window_seconds: float = 10.0):
        self.window_seconds = window_seconds
        self._buf: deque[StateT] = deque()

    def push(self, s: StateT) -> None:
        self._buf.append(s)
        cutoff = s.t - self.window_seconds
        while self._buf and self._buf[0].t < cutoff:
            self._buf.popleft()

    def latest(self) -> Optional[StateT]:
        return self._buf[-1] if self._buf else None

    def window(self, t_start: float, t_end: float) -> list[StateT]:
        return [s for s in self._buf if t_start <= s.t <= t_end]

    def __len__(self) -> int:
        return len(self._buf)
