"""Smoke test: render a rewind card from a fake System 2 response.

Writes the resulting image to /tmp/rewind_test.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the repo: `python scripts/smoke_system2_render.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from mac.frame_data import FrameData
from mac.rewind_card import render_card


def main() -> None:
    fd = FrameData.load("data/frame_data.json", "data/hitboxes.json")

    response = {
        "summary": "Joker landed gun on p2's shield; spotdodge would have flipped the exchange.",
        "chosen_action": {
            "player": "p1",
            "action_label": "shield",
            "frame_t": 1.0,
            "reasoning": "Held shield as Toon Link committed to fsmash from medium range.",
        },
        "counterfactual_action": {
            "action_label": "spotdodge",
            "rationale": "Spotdodge would have avoided the active hitbox and opened a punish.",
            "qualitative_likelihood": "likely",
        },
        "frame_data_citations": [
            {"character": "toon_link", "move": "fsmash", "stat": "startup_f", "value": "15"},
            {"character": "toon_link", "move": "fsmash", "stat": "endlag_f", "value": "40"},
            {"character": "joker", "move": "jab", "stat": "startup_f", "value": "4"},
            {"character": "joker", "move": "ftilt", "stat": "startup_f", "value": "8"},
        ],
    }

    char_map = {"p1": "joker", "p2": "toon_link"}

    # Fake keyframes: 4 solid-color images at increasing timestamps.
    colors = [(40, 40, 80), (40, 80, 40), (80, 40, 40), (80, 80, 40)]
    keyframes: list[tuple[float, np.ndarray]] = []
    for i, c in enumerate(colors):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[:] = c
        keyframes.append((float(i) * 0.5, img))

    card = render_card(keyframes, response, fd, char_map)
    out = "/tmp/rewind_test.png"
    cv2.imwrite(out, card)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
