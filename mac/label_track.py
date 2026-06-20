"""Precomputed label track for prerecorded-clip playback.

For a recorded MP4 we don't need live inference: we run the (accurate, slow)
72B over the clip once offline, save a track of states keyed by video time, and
at playback look up the state for the displayed frame. Result: 72B accuracy +
perfect frame/label sync + smooth full-speed playback, with zero live latency.

Track JSON format:
    {"meta": {...}, "entries": [{"video_t": 11.0, "p1": {...}, "p2": {...}}, ...]}
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

from mac.dispatcher import S1Out


class LabelTrack:
    def __init__(self, entries: list[dict]):
        self.entries = sorted(entries, key=lambda e: e["video_t"])
        self._ts = [e["video_t"] for e in self.entries]

    @classmethod
    def load(cls, path: str) -> "LabelTrack":
        data = json.loads(Path(path).read_text())
        entries = data["entries"] if isinstance(data, dict) else data
        return cls(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def at(self, video_t: float) -> S1Out | None:
        """State for the frame at ``video_t`` — the most recent entry whose
        video_t is <= the query (holds the last label until the next sample)."""
        if not self.entries:
            return None
        i = bisect.bisect_right(self._ts, video_t) - 1
        if i < 0:
            i = 0
        e = self.entries[i]
        return S1Out(p1=e["p1"], p2=e["p2"], t=e["video_t"])
