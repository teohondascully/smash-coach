"""Frame-data + hitbox loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class Move(BaseModel):
    name: str
    startup_f: Optional[int] = None
    active_f: Optional[list[int]] = None  # [first, last]
    endlag_f: Optional[int] = None
    landing_lag_f: Optional[int] = None
    shield_advantage: Optional[int] = None
    on_hit_advantage: Optional[int] = None
    range_estimate: str = "medium"
    category: str = "ground"


class HitboxCircle(BaseModel):
    dx: float
    dy: float
    radius: float
    active_frames: list[int]  # [first, last]


class FrameData:
    def __init__(
        self,
        moves: dict[str, dict[str, Move]],
        hitboxes: dict[str, dict[str, list[HitboxCircle]]],
    ):
        self.moves = moves
        self.hitboxes_data = hitboxes

    @classmethod
    def load(cls, frame_data_path: str, hitboxes_path: str) -> "FrameData":
        raw = json.loads(Path(frame_data_path).read_text())
        moves = {
            char: {m["name"]: Move(**m) for m in lst} for char, lst in raw.items()
        }
        hb_path = Path(hitboxes_path)
        hb_raw: dict = {}
        if hb_path.exists():
            try:
                hb_raw = json.loads(hb_path.read_text())
            except json.JSONDecodeError:
                hb_raw = {}
        hitboxes = {
            char: {
                mv: [HitboxCircle(**c) for c in circles]
                for mv, circles in mv_dict.items()
            }
            for char, mv_dict in hb_raw.items()
        }
        return cls(moves, hitboxes)

    def move(self, char: str, name: str) -> Move:
        return self.moves[char][name]

    def hitboxes(self, char: str, name: str, frame_in_move: int) -> list[HitboxCircle]:
        char_hb = self.hitboxes_data.get(char, {})
        circles = char_hb.get(name, [])
        return [
            c
            for c in circles
            if c.active_frames[0] <= frame_in_move <= c.active_frames[1]
        ]
