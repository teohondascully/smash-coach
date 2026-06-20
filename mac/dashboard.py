"""Debug/monitoring dashboard for the Smash Coach.

Renders a 1280x720 BGR composite with four 640x360 quadrants:

  +------------------+------------------+
  | LIVE HUD         | RAW CAPTURE      |
  +------------------+------------------+
  | STATE TEXT       | METRICS + LOGS   |
  +------------------+------------------+

The dashboard is enabled when DEBUG env is truthy or toggled at runtime via
the 'd' key in mac.main. All inputs are tolerated as None and rendered as
graceful "(waiting...)" placeholders.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from mac.dispatcher import S1Out
from mac.state import StateT

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.5
THICK = 1
LINE_H = 22

WHITE = (255, 255, 255)
GRAY = (170, 170, 170)
YELLOW = (0, 220, 220)
RED = (60, 60, 230)
GREEN = (120, 220, 120)
CYAN = (220, 220, 60)
BORDER = (255, 255, 255)

W, H = 1280, 720
QW, QH = 640, 360


def _put(img, text: str, org, color=WHITE, scale=SCALE, thick=THICK) -> None:
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _rel_age(now: float, ts: Optional[float]) -> str:
    if ts is None:
        return "idle"
    dt = max(0.0, now - ts)
    if dt < 1.0:
        return f"{int(dt * 1000)}ms ago"
    return f"{dt:.1f}s ago"


def _log_color(msg: str):
    lower = msg.lower()
    if any(k in lower for k in ("error", "crashed", "fail", "exception")):
        return RED
    if any(k in lower for k in ("trigger", "warn", "warning")):
        return YELLOW
    return GRAY


class Dashboard:
    def __init__(self, max_log_lines: int = 12):
        self.max_log_lines = max_log_lines
        self._logs: deque[tuple[float, str]] = deque(maxlen=max_log_lines)

    def log(self, msg: str) -> None:
        self._logs.append((time.monotonic(), str(msg)))

    # ---------- quadrant renderers ----------

    def _panel(self, title: str) -> np.ndarray:
        p = np.zeros((QH, QW, 3), dtype=np.uint8)
        _put(p, title, (10, 22), WHITE, scale=0.6, thick=1)
        cv2.line(p, (0, 28), (QW, 28), (80, 80, 80), 1)
        return p

    def _render_state(self, state: Optional[StateT]) -> np.ndarray:
        panel = self._panel("STATE")
        if state is None:
            _put(panel, "(waiting for first state)", (10, 60), GRAY)
            return panel
        y = 56
        _put(panel, f"t = {state.t:.2f}s", (10, y), CYAN); y += LINE_H
        d = state.damage
        _put(panel, f"damage  p1={d.get('p1', 0):6.1f}%   p2={d.get('p2', 0):6.1f}%",
             (10, y), WHITE); y += LINE_H
        s = state.stocks
        _put(panel, f"stocks  p1={s.get('p1', 0)}   p2={s.get('p2', 0)}",
             (10, y), WHITE); y += LINE_H
        der = state.derived
        _put(panel,
             f"distance={der.distance:.1f}  facing={der.relative_facing}  "
             f"punish={der.active_punish_window_for or '-'}",
             (10, y), GRAY); y += LINE_H
        y += 6
        for who in ("p1", "p2"):
            a = state.actions.get(who)
            pos = state.positions.get(who)
            intent = state.intent.get(who, "?")
            color = (255, 200, 0) if who == "p1" else (0, 200, 255)
            _put(panel, f"[{who}]", (10, y), color); y += LINE_H
            if a is not None:
                onset_age = _rel_age(state.t, a.onset_estimate_t) if a.onset_estimate_t else "-"
                _put(panel,
                     f"  {a.label}  phase={a.phase}  conf={a.confidence:.2f}  "
                     f"onset={onset_age}",
                     (10, y), WHITE); y += LINE_H
            if pos is not None:
                _put(panel,
                     f"  pos=({pos.x:.0f},{pos.y:.0f}) face={pos.facing} "
                     f"air={int(pos.airborne)} v=({pos.vx:.1f},{pos.vy:.1f})",
                     (10, y), GRAY); y += LINE_H
            _put(panel, f"  intent={intent}", (10, y), GRAY); y += LINE_H
            y += 4
        return panel

    def _render_metrics(
        self,
        metrics: dict,
        last_s1_t: Optional[float],
        last_s2_resp: Optional[dict],
        last_s2_resp_t: Optional[float],
        s2_pending: bool,
        trigger_count: int,
        error_count: int,
        now: float,
    ) -> np.ndarray:
        panel = self._panel("METRICS / LOGS")
        y = 56
        fps = metrics.get("fps", 0.0)
        _put(panel, f"fps={fps:6.2f}", (10, y), CYAN); y += LINE_H
        # per-stage ms
        for k in ("tier0", "tier1", "s1_wait", "hud", "draw"):
            if k in metrics:
                _put(panel, f"  {k:<10}{metrics[k]:6.2f} ms", (10, y), WHITE)
                y += LINE_H
        y += 4
        _put(panel, f"last_s1: {_rel_age(now, last_s1_t)}", (10, y), WHITE); y += LINE_H
        s2_state = "PENDING" if s2_pending else _rel_age(now, last_s2_resp_t)
        _put(panel, f"last_s2: {s2_state}", (10, y),
             YELLOW if s2_pending else WHITE); y += LINE_H
        if last_s2_resp is not None:
            summ = str(last_s2_resp.get("summary", ""))[:62]
            _put(panel, f"  s2.summary: {summ}", (10, y), GRAY); y += LINE_H
        _put(panel, f"triggers={trigger_count}   errors={error_count}",
             (10, y), GREEN if error_count == 0 else RED); y += LINE_H

        # logs in bottom portion - paint a clean band first
        log_top = max(y + 4, QH // 2 + 8)
        cv2.rectangle(panel, (0, log_top - 4), (QW, QH), (0, 0, 0), -1)
        cv2.line(panel, (0, log_top - 4), (QW, log_top - 4), (80, 80, 80), 1)
        _put(panel, "logs (oldest -> newest):", (10, log_top + 14), GRAY, scale=0.45)
        ly = log_top + 14 + LINE_H
        logs = list(self._logs)
        max_visible = max(1, (QH - ly - 4) // LINE_H)
        for ts, msg in logs[-max_visible:]:
            age = now - ts
            stamp = f"-{age:4.1f}s "
            line = (stamp + msg)[:80]
            _put(panel, line, (10, ly), _log_color(msg))
            ly += LINE_H
        return panel

    # ---------- main render ----------

    def render(
        self,
        live_hud: np.ndarray,
        raw_capture: np.ndarray,
        state: Optional[StateT],
        last_s1_out: Optional[S1Out],
        last_s1_t: Optional[float],
        last_s2_resp: Optional[dict],
        s2_pending: bool,
        metrics: dict,
        trigger_count: int,
        error_count: int,
        now: float,
        last_s2_resp_t: Optional[float] = None,
    ) -> np.ndarray:
        out = np.zeros((H, W, 3), dtype=np.uint8)

        # Top-left: live HUD
        tl = _fit(live_hud, QW, QH)
        out[0:QH, 0:QW] = tl
        _put(out, "LIVE HUD", (10, 22), WHITE, scale=0.6)

        # Top-right: raw capture
        tr = _fit(raw_capture, QW, QH)
        out[0:QH, QW:W] = tr
        _put(out, "RAW CAPTURE", (QW + 10, 22), WHITE, scale=0.6)

        # Bottom-left: state
        bl = self._render_state(state)
        # show last_s1 footer
        if last_s1_out is not None:
            _put(bl,
                 f"s1: p1={last_s1_out.p1.get('action_label','?')} "
                 f"p2={last_s1_out.p2.get('action_label','?')}",
                 (10, QH - 10), CYAN, scale=0.45)
        else:
            _put(bl, "s1: (no response yet)", (10, QH - 10), GRAY, scale=0.45)
        out[QH:H, 0:QW] = bl

        # Bottom-right: metrics + logs
        br = self._render_metrics(
            metrics, last_s1_t, last_s2_resp, last_s2_resp_t,
            s2_pending, trigger_count, error_count, now,
        )
        out[QH:H, QW:W] = br

        # Borders
        cv2.line(out, (QW, 0), (QW, H), BORDER, 1)
        cv2.line(out, (0, QH), (W, QH), BORDER, 1)
        cv2.rectangle(out, (0, 0), (W - 1, H - 1), BORDER, 1)

        return out
