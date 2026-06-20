"""Action onset tracking from System 1 action label stream."""
from __future__ import annotations

from typing import Optional


class OnsetTracker:
    """Records the timestamp at which a player's current action began.

    Onset is reset whenever the action label changes. ``frame_in_move`` converts
    elapsed wall-time since onset into a frame index (assuming 60 Hz game).
    """

    def __init__(self) -> None:
        self._last_action: dict[str, str] = {}
        self._onset_t: dict[str, float] = {}

    def update(self, player: str, action: str, t: float) -> None:
        if self._last_action.get(player) != action:
            self._onset_t[player] = t
            self._last_action[player] = action

    def onset(self, player: str) -> Optional[float]:
        return self._onset_t.get(player)

    def frame_in_move(self, player: str, now: float) -> int:
        o = self._onset_t.get(player)
        if o is None:
            return 0
        return int(max(0, (now - o) * 60))
