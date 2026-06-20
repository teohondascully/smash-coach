"""Per-player majority-vote smoother for noisy VLM action labels.

The HUD label rendered from raw S1 output flickers a lot when the model
is between two equally-plausible actions (e.g. ftilt vs jab during a
fast sequence). A 3-frame majority vote stabilizes the displayed label
without delaying real transitions perceptibly.
"""
from __future__ import annotations

from collections import Counter, deque


class LabelSmoother:
    def __init__(self, k: int = 3):
        self.k = k
        self._hist: dict[str, deque[str]] = {
            "p1": deque(maxlen=k),
            "p2": deque(maxlen=k),
        }

    def update(self, who: str, label: str) -> str:
        h = self._hist[who]
        h.append(label)
        return Counter(h).most_common(1)[0][0]
