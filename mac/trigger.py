"""Event trigger detector for System 2 dispatch.

Watches a stream of StateT and fires on stock loss or significant damage spikes
in a short window, with a cooldown to avoid duplicate fires.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mac.state import StateT


@dataclass
class TriggerEvent:
    kind: str  # "stock_loss" | "exchange"
    who: Optional[str]
    t: float


class TriggerDetector:
    def __init__(
        self,
        damage_delta: float = 30.0,
        damage_window_s: float = 2.0,
        cooldown_s: float = 5.0,
    ):
        self.damage_delta = damage_delta
        self.damage_window_s = damage_window_s
        self.cooldown_s = cooldown_s
        self._history: list[StateT] = []
        self._last_fire_t: float = -1e9

    def check(self, s: StateT) -> Optional[TriggerEvent]:
        prev = self._history[-1] if self._history else None
        self._history.append(s)
        # Keep enough history for the damage window (plus a small cushion).
        keep_from = s.t - max(self.damage_window_s, 5.0)
        self._history = [h for h in self._history if h.t >= keep_from]

        if s.t - self._last_fire_t < self.cooldown_s:
            return None

        # Stock loss takes precedence.
        if prev is not None:
            for who in ("p1", "p2"):
                if s.stocks.get(who, 0) < prev.stocks.get(who, 0):
                    self._last_fire_t = s.t
                    return TriggerEvent("stock_loss", who, s.t)

        # Damage spike over window.
        window = [h for h in self._history if h.t >= s.t - self.damage_window_s]
        if len(window) >= 2:
            base = window[0]
            for who in ("p1", "p2"):
                if s.damage.get(who, 0.0) - base.damage.get(who, 0.0) >= self.damage_delta:
                    self._last_fire_t = s.t
                    return TriggerEvent("exchange", who, s.t)
        return None
