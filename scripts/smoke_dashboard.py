"""Smoke test for the debug Dashboard.

Constructs fake StateT / S1Out / S2 response and renders one composite frame
to /tmp/dashboard_test.png so you can eyeball the layout without a live
capture device or GPU node.

Run:
    uv run python scripts/smoke_dashboard.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Allow running as `uv run python scripts/smoke_dashboard.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from mac.dashboard import Dashboard
from mac.dispatcher import S1Out
from mac.state import ActionState, PlayerState, StateT


def make_fake_image(label: str, color: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img[:] = color
    cv2.putText(
        img, label, (60, 540), cv2.FONT_HERSHEY_SIMPLEX, 4.0,
        (255, 255, 255), 8, cv2.LINE_AA,
    )
    cv2.rectangle(img, (40, 40), (1880, 1040), (255, 255, 255), 6)
    return img


def main() -> None:
    dash = Dashboard()

    now_t_game = 12.345  # game-time t (in StateT)
    state = StateT(
        t=now_t_game,
        damage={"p1": 42.0, "p2": 88.0},
        stocks={"p1": 1, "p2": 1},
        positions={
            "p1": PlayerState(x=480.0, y=540.0, facing="right",
                              airborne=False, vx=1.2, vy=0.0),
            "p2": PlayerState(x=1340.0, y=520.0, facing="left",
                              airborne=True, vx=-0.5, vy=-2.1),
        },
        actions={
            "p1": ActionState(label="shield", phase="active",
                              confidence=0.71,
                              onset_estimate_t=now_t_game - 0.2),
            "p2": ActionState(label="fsmash", phase="startup",
                              confidence=0.88,
                              onset_estimate_t=now_t_game - 0.15),
        },
        intent={"p1": "neutral", "p2": "pressuring"},
    )

    last_s1 = S1Out(
        p1={"action_label": "shield", "phase": "active",
            "confidence": 0.71, "intent": "neutral"},
        p2={"action_label": "fsmash", "phase": "startup",
            "confidence": 0.88, "intent": "pressuring"},
        t=now_t_game,
    )

    last_s2_resp = {
        "summary": ("Joker shielded a Toon Link fsmash on stage; "
                    "a spotdodge would have opened a punish window."),
        "chosen_action": {
            "player": "p1", "action_label": "shield",
            "frame_t": now_t_game - 0.2,
            "reasoning": "Held shield through startup, no OoS punish.",
        },
        "counterfactual_action": {
            "action_label": "spotdodge",
            "rationale": ("Spotdodge on frame 14 of fsmash startup gives "
                          "a 9-frame punish window after recovery."),
            "qualitative_likelihood": "likely",
        },
        "frame_data_citations": [
            {"character": "toon_link", "move": "fsmash",
             "stat": "startup_f", "value": "16"},
            {"character": "toon_link", "move": "fsmash",
             "stat": "endlag_f", "value": "38"},
            {"character": "joker", "move": "spotdodge",
             "stat": "active_f", "value": "[3,20]"},
        ],
    }

    now_mono = time.monotonic()
    metrics = {
        "fps": 58.4,
        "tier0": 1.2,
        "tier1": 2.7,
        "s1_wait": 0.4,
        "hud": 3.1,
        "draw": 0.8,
    }

    dash.log("[main] starting | CAP_DEV=0 hz=7.0")
    dash.log("[main] system1 first response 312ms")
    dash.log("[main] trigger: exchange who=p1 t=12.10 keyframes=5")
    dash.log("[main] warn: ocr.damage low-contrast frame")
    dash.log("[main] system2 task crashed: connection refused")
    dash.log("[main] rewind shown | summary='Joker shielded a Toon Link...'")
    dash.log("[main] fps=58.4 avg ms/frame tier0=1.2 tier1=2.7")

    live_hud = make_fake_image("LIVE HUD (composed)", (40, 60, 30))
    raw_capture = make_fake_image("RAW CAPTURE", (30, 30, 70))

    img = dash.render(
        live_hud=live_hud,
        raw_capture=raw_capture,
        state=state,
        last_s1_out=last_s1,
        last_s1_t=now_mono - 0.312,
        last_s2_resp=last_s2_resp,
        s2_pending=False,
        metrics=metrics,
        trigger_count=3,
        error_count=1,
        now=now_mono,
        last_s2_resp_t=now_mono - 4.2,
    )

    out_path = "/tmp/dashboard_test.png"
    ok = cv2.imwrite(out_path, img)
    if not ok:
        raise SystemExit(f"cv2.imwrite failed for {out_path}")
    size = os.path.getsize(out_path)
    print(f"wrote {out_path} ({size} bytes, shape={img.shape})")


if __name__ == "__main__":
    main()
