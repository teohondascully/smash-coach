"""OpenCV HUD overlay renderer.

Draws damage %, action labels, approximated hitboxes, and threat-zone radii on
top of a captured frame. Unknown action labels (not in the frame-data table)
are skipped silently rather than crashing the live loop.
"""
from __future__ import annotations

import cv2

from mac.frame_data import FrameData
from mac.onset import OnsetTracker
from mac.state import StateT

_THREAT_RADIUS = {"short": 40, "medium": 70, "long": 110}


class HUD:
    def __init__(self, fd: FrameData, onset: OnsetTracker):
        self.fd = fd
        self.onset = onset

    def draw(self, img, s: StateT, now: float, char_map: dict):
        out = img.copy()
        # Damage is intentionally NOT drawn — the game already shows it. We read
        # it into state for coaching/triggers, not to echo it back on screen.
        self._draw_action_labels(out, s, char_map)
        self._draw_hitboxes(out, s, now, char_map)
        self._draw_threat_zones(out, s, char_map)
        return out

    def _draw_damage(self, img, s: StateT) -> None:
        h, w = img.shape[:2]
        cv2.putText(
            img, f"P1 {s.damage['p1']:.1f}%",
            (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3,
        )
        cv2.putText(
            img, f"P2 {s.damage['p2']:.1f}%",
            (w - 340, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3,
        )

    def _draw_action_labels(self, img, s: StateT, char_map: dict) -> None:
        # Fixed, frame-relative anchors (P1 top-left, P2 top-right) so labels are
        # always visible regardless of bbox detection or capture resolution.
        # (The old bbox-anchored positions were calibrated for 1080p and pushed
        # P2's label off-screen on a 720p source.)
        h, w = img.shape[:2]
        for who, color, anchor in (
            ("p1", (255, 200, 0), "left"),
            ("p2", (0, 200, 255), "right"),
        ):
            a = s.actions[who]
            name = char_map.get(who, who)
            txt = f"{name}: {a.label}"
            if a.phase not in ("neutral", "unknown"):
                txt += f" [{a.phase}]"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            y = 44
            x = 30 if anchor == "left" else max(10, w - tw - 30)
            cv2.rectangle(img, (x - 12, y - th - 14), (x + tw + 12, y + 14),
                          (0, 0, 0), -1)
            cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    def _draw_hitboxes(self, img, s: StateT, now: float, char_map: dict) -> None:
        for who in ("p1", "p2"):
            char = char_map[who]
            mv = s.actions[who].label
            try:
                self.fd.move(char, mv)
            except KeyError:
                continue
            fim = self.onset.frame_in_move(who, now)
            try:
                circles = self.fd.hitboxes(char, mv, fim)
            except KeyError:
                continue
            p = s.positions[who]
            sign = 1 if p.facing == "right" else -1
            for c in circles:
                cx = int(p.x + sign * c.dx)
                cy = int(p.y + c.dy)
                cv2.circle(img, (cx, cy), int(c.radius), (0, 0, 255), 2)

    def _draw_threat_zones(self, img, s: StateT, char_map: dict) -> None:
        for who, color in (("p1", (0, 100, 255)), ("p2", (100, 0, 255))):
            char = char_map[who]
            mv = s.actions[who].label
            try:
                m = self.fd.move(char, mv)
            except KeyError:
                continue
            r = _THREAT_RADIUS.get(m.range_estimate, 60)
            p = s.positions[who]
            cv2.circle(img, (int(p.x), int(p.y)), r, color, 1)
