"""Rewind card renderer for System 2 counterfactual responses.

Implements Task 5.4 of the smash-coach plan. Produces a 1280x720 BGR image
summarizing what happened, the chosen alternative, and frame-data citations.
"""
from __future__ import annotations

import cv2
import numpy as np

from mac.frame_data import FrameData
from mac.scorer import score_counterfactual


_RED = (60, 60, 220)        # BGR
_GREEN = (80, 200, 90)
_YELLOW = (60, 220, 230)
_GRAY = (180, 180, 180)
_WHITE = (240, 240, 240)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put(img: np.ndarray, text: str, org: tuple[int, int], color, scale: float = 0.7, thick: int = 2) -> None:
    cv2.putText(img, text, org, _FONT, scale, color, thick, cv2.LINE_AA)


def render_card(
    keyframes: list[tuple[float, np.ndarray]],
    response: dict,
    fd: FrameData,
    char_map: dict[str, str],
) -> np.ndarray:
    """Render a 1280x720 BGR rewind card.

    Any rendering errors are caught field-by-field so a partial response will
    still produce a usable image (never crash the demo).
    """
    card = np.zeros((720, 1280, 3), dtype=np.uint8)

    # --- Top strip: up to 4 most recent keyframes (y 20..300) -----------------
    try:
        recent = keyframes[-4:] if keyframes else []
        n = max(1, len(recent))
        if recent:
            thumb_w = 1280 // n
            thumb_h = 300 - 20
            for i, (t, frame) in enumerate(recent):
                try:
                    resized = cv2.resize(frame, (thumb_w, thumb_h))
                    x0 = i * thumb_w
                    card[20:20 + thumb_h, x0:x0 + thumb_w] = resized
                    _put(card, f"t={t:.2f}", (x0 + 8, 40), _WHITE, 0.5, 1)
                except Exception:
                    _put(card, "(thumb error)", (i * thumb_w + 8, 160), _RED, 0.6, 1)
    except Exception:
        _put(card, "(rendering error: keyframes)", (20, 160), _RED)

    # --- Summary --------------------------------------------------------------
    try:
        summary = str(response.get("summary", ""))[:80]
        _put(card, summary, (20, 340), _WHITE, 0.7, 2)
    except Exception:
        _put(card, "(rendering error: summary)", (20, 340), _RED)

    # --- Chosen action --------------------------------------------------------
    chosen = response.get("chosen_action", {}) or {}
    counterfactual = response.get("counterfactual_action", {}) or {}

    try:
        _put(card, f"You did: {chosen['action_label']}", (20, 400), _RED, 0.8, 2)
    except Exception:
        _put(card, "(rendering error: chosen_action)", (20, 400), _RED, 0.7, 2)

    # --- Counterfactual -------------------------------------------------------
    try:
        alt_label = counterfactual["action_label"]
        likelihood = counterfactual["qualitative_likelihood"]
        _put(card, f"Alt: {alt_label} ({likelihood})", (20, 440), _GREEN, 0.8, 2)
    except Exception:
        _put(card, "(rendering error: counterfactual_action)", (20, 440), _RED, 0.7, 2)

    # --- Deterministic scoring ------------------------------------------------
    try:
        defender_who = chosen["player"]
        attacker_who = "p2" if defender_who == "p1" else "p1"
        score = score_counterfactual(
            fd,
            attacker_char=char_map[attacker_who],
            attacker_move=chosen["action_label"],
            defender_char=char_map[defender_who],
            defender_response=counterfactual["action_label"],
        )
        n_frames = int(score.get("punish_window_frames", 0) or 0)
        if n_frames > 0:
            _put(
                card,
                f"Would have opened a {n_frames}-frame punish window",
                (20, 490),
                _YELLOW,
                0.75,
                2,
            )
    except Exception:
        _put(card, "(rendering error: scoring)", (20, 490), _RED, 0.7, 2)

    # --- Citations (up to 4) --------------------------------------------------
    try:
        citations = response.get("frame_data_citations", []) or []
        for i, c in enumerate(citations[:4]):
            try:
                line = f"  {c['character']} {c['move']}: {c['stat']}={c['value']}"
                _put(card, line, (20, 540 + i * 28), _GRAY, 0.6, 1)
            except Exception:
                _put(card, "  (citation error)", (20, 540 + i * 28), _RED, 0.6, 1)
    except Exception:
        _put(card, "(rendering error: citations)", (20, 540), _RED, 0.7, 1)

    return card


def show_card(card: np.ndarray, duration_s: float = 6.0) -> None:
    # NOTE: this call is BLOCKING for ``duration_s`` seconds. main.py's
    # integrator may want to wrap this in a thread / asyncio.to_thread to
    # avoid stalling the capture/state loop.
    cv2.imshow("rewind-card", card)
    cv2.waitKey(int(duration_s * 1000))
    cv2.destroyWindow("rewind-card")
