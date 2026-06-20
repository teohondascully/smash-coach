"""Smash Coach orchestrator.

Env vars:
  CAP_DEV         capture device index (default 0)
  S1_URL          System 1 endpoint   (default http://localhost:8001/infer)
  S2_URL          System 2 endpoint   (default http://localhost:8002/counterfactual)
  S1_HZ           System 1 poll rate  (default 7.0)
  REWIND_SECS     rewind card visibility duration (default 6.0)
  TRIGGER_DELTA   damage delta to fire System 2 (default 30.0)
  TRIGGER_COOLDOWN seconds between consecutive triggers (default 5.0)

Graceful degradation:
  - If S1 URL unreachable: HUD shows Tier 0/1 (damage, bboxes), action labels stay "neutral".
  - If S2 URL unreachable: trigger fires but no rewind card shown.
  - If capture device open fails: prints hint and exits 0.

Press 'q' in the smash-coach window to quit.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque

import cv2
import numpy as np

from mac.capture import Capture
from mac.dashboard import Dashboard
from mac.dispatcher import S1Out, System1Client
from mac.frame_data import FrameData
from mac.hud import HUD
from mac.onset import OnsetTracker
from mac.rewind_card import render_card
from mac.smoother import LabelSmoother
from mac.state import ActionState, PlayerState, StateBuffer, StateT
from mac.system2_client import System2Client, select_keyframes
from mac.tier0_ocr import Tier0
from mac.tier1_cv import detect as detect_bboxes
from mac.trigger import TriggerDetector

CHAR_MAP = {
    "p1": os.getenv("P1_CHAR", "toon_link"),
    "p2": os.getenv("P2_CHAR", "ike"),
}


class RewindCardWindow:
    """Non-blocking rewind card display. Pumped by main loop's cv2.waitKey."""

    def __init__(self, duration_s: float = 6.0):
        self.duration_s = duration_s
        self._shown_at = 0.0
        self._visible = False

    def show(self, card: np.ndarray) -> None:
        cv2.imshow("rewind-card", card)
        self._shown_at = time.monotonic()
        self._visible = True

    def tick(self, now: float) -> None:
        if self._visible and now - self._shown_at > self.duration_s:
            cv2.destroyWindow("rewind-card")
            self._visible = False


def _position_from_bbox(bb, default_x: float) -> PlayerState:
    if bb is None:
        return PlayerState(
            x=default_x, y=500.0, facing="right", airborne=False, vx=0.0, vy=0.0
        )
    return PlayerState(
        x=bb.cx, y=bb.cy, facing="right", airborne=False, vx=0.0, vy=0.0
    )


def _build_actions_intent(last_s1: S1Out | None, onset: OnsetTracker, t: float):
    if last_s1 is None:
        return (
            {who: ActionState(label="neutral") for who in ("p1", "p2")},
            {"p1": "neutral", "p2": "neutral"},
        )
    actions = {
        who: ActionState(
            label=getattr(last_s1, who)["action_label"],
            phase=getattr(last_s1, who)["phase"],
            confidence=getattr(last_s1, who)["confidence"],
            onset_estimate_t=onset.onset(who) or t,
        )
        for who in ("p1", "p2")
    }
    intent = {who: getattr(last_s1, who)["intent"] for who in ("p1", "p2")}
    return actions, intent


async def run() -> None:
    cap_dev = int(os.getenv("CAP_DEV", "0"))
    s1_url = os.getenv("S1_URL", "http://localhost:8001/infer")
    s2_url = os.getenv("S2_URL", "http://localhost:8002/counterfactual")
    s1_hz = float(os.getenv("S1_HZ", "7.0"))
    rewind_secs = float(os.getenv("REWIND_SECS", "6.0"))
    trigger_delta = float(os.getenv("TRIGGER_DELTA", "30.0"))
    trigger_cooldown = float(os.getenv("TRIGGER_COOLDOWN", "5.0"))
    debug_on = os.getenv("DEBUG", "").lower() in ("1", "true", "yes", "on")
    record_path = os.getenv("RECORD_PATH", "")  # if set, save HUD output to MP4
    dashboard = Dashboard()

    def dlog(msg: str) -> None:
        print(msg)
        dashboard.log(msg)

    try:
        cap = Capture(device_index=cap_dev)
    except RuntimeError as e:
        print(f"[main] capture failed: {e}")
        print("[main] tip: try CAP_DEV=1 or CAP_DEV=2; "
              "check System Settings > Privacy & Security > Camera")
        return

    ocr = Tier0("data/ui_regions.json")
    fd = FrameData.load("data/frame_data.json", "data/hitboxes.json")
    onset = OnsetTracker()
    hud = HUD(fd, onset)
    buf = StateBuffer(window_seconds=10.0)
    s1 = System1Client(url=s1_url, hz=s1_hz)
    s2 = System2Client(url=s2_url)
    trigger = TriggerDetector(
        damage_delta=trigger_delta,
        damage_window_s=2.0,
        cooldown_s=trigger_cooldown,
    )
    rewind = RewindCardWindow(duration_s=rewind_secs)
    smoother = LabelSmoother(k=3)
    # Optional MP4 recording of the HUD output — bulletproof demo backup.
    # Initialized lazily on the first frame so the codec sees a real size.
    recorder: cv2.VideoWriter | None = None

    raw_frames: deque[tuple[float, np.ndarray]] = deque(maxlen=600)
    last_s1: S1Out | None = None
    last_s1_t: float | None = None
    last_s2_resp: dict | None = None
    last_s2_resp_t: float | None = None
    pending_s2: asyncio.Task | None = None
    last_trigger_keyframes: list[tuple[float, np.ndarray]] = []
    frames_since_log = 0
    timing = {"tier0": 0.0, "tier1": 0.0, "s1_wait": 0.0, "hud": 0.0, "draw": 0.0}
    window_start_t = time.monotonic()
    last_metrics: dict[str, float] = {
        "fps": 0.0, "tier0": 0.0, "tier1": 0.0, "s1_wait": 0.0,
        "hud": 0.0, "draw": 0.0,
    }
    trigger_count = 0
    error_count = 0

    dlog(f"[main] starting | CAP_DEV={cap_dev} S1_URL={s1_url} S2_URL={s2_url} hz={s1_hz}")
    dlog("[main] press 'q' in the smash-coach window to quit; 'd' toggles dashboard")
    if debug_on:
        dlog("[main] DEBUG=on; dashboard window enabled")

    for frame in cap.frames():
        t = frame.t
        raw_frames.append((t, frame.img.copy()))

        # Tier 0 (stocks only — damage now comes from System 1's VLM output)
        t0 = time.monotonic()
        try:
            st1 = ocr.stocks(frame.img, "p1")
            st2 = ocr.stocks(frame.img, "p2")
        except Exception as e:
            error_count += 1
            dlog(f"[main] tier0 stocks error: {e}")
            st1, st2 = 1, 1
        timing["tier0"] += time.monotonic() - t0

        # Tier 1
        t0 = time.monotonic()
        try:
            bb = detect_bboxes(frame.img)
        except Exception as e:
            error_count += 1
            dlog(f"[main] tier1 detect error: {e}")
            bb = {"p1": None, "p2": None}
        positions = {
            "p1": _position_from_bbox(bb["p1"], 500.0),
            "p2": _position_from_bbox(bb["p2"], 1400.0),
        }
        timing["tier1"] += time.monotonic() - t0

        # System 1 (rate-limited inside dispatcher)
        t0 = time.monotonic()
        out = await s1.maybe_infer(frame.img, t)
        timing["s1_wait"] += time.monotonic() - t0
        if out is not None:
            # Smooth the raw labels through a 3-frame majority vote so the
            # HUD doesn't flicker between equally-plausible actions.
            out.p1["action_label"] = smoother.update("p1", out.p1["action_label"])
            out.p2["action_label"] = smoother.update("p2", out.p2["action_label"])
            last_s1 = out
            last_s1_t = time.monotonic()
            onset.update("p1", out.p1["action_label"], t)
            onset.update("p2", out.p2["action_label"], t)

        actions, intent = _build_actions_intent(last_s1, onset, t)
        # Damage comes from S1 now. Until the first S1 response lands, default
        # to 0 — graceful, the HUD just shows 0% momentarily.
        d1 = float(last_s1.p1.get("damage_pct", 0)) if last_s1 else 0.0
        d2 = float(last_s1.p2.get("damage_pct", 0)) if last_s1 else 0.0
        s = StateT(
            t=t,
            damage={"p1": d1, "p2": d2},
            stocks={"p1": st1, "p2": st2},
            positions=positions,
            actions=actions,
            intent=intent,
        )
        buf.push(s)

        # Trigger
        ev = trigger.check(s)
        if ev is not None and (pending_s2 is None or pending_s2.done()):
            trigger_count += 1
            window_states = buf.window(ev.t - 5.0, ev.t)
            last_trigger_keyframes = select_keyframes(
                list(raw_frames), window_states, max_n=8
            )
            dlog(f"[main] trigger: {ev.kind} who={ev.who} t={ev.t:.2f} "
                 f"window_states={len(window_states)} "
                 f"keyframes={len(last_trigger_keyframes)}")
            pending_s2 = asyncio.create_task(
                s2.request(ev, window_states, last_trigger_keyframes)
            )

        # System 2 response handling
        if pending_s2 is not None and pending_s2.done():
            try:
                resp = pending_s2.result()
            except Exception as e:
                error_count += 1
                dlog(f"[main] system2 task crashed: {e}")
                resp = None
            pending_s2 = None
            if resp is not None:
                last_s2_resp = resp
                last_s2_resp_t = time.monotonic()
                try:
                    card = render_card(last_trigger_keyframes, resp, fd, CHAR_MAP)
                    rewind.show(card)
                    dlog(f"[main] rewind shown | "
                         f"summary={resp.get('summary','')[:60]!r}")
                except Exception as e:
                    error_count += 1
                    dlog(f"[main] rewind render failed: {e}")

        rewind.tick(time.monotonic())

        # HUD compose
        t0 = time.monotonic()
        try:
            composed = hud.draw(frame.img, s, t, CHAR_MAP)
        except Exception as e:
            error_count += 1
            dlog(f"[main] hud draw error: {e}")
            composed = frame.img
        timing["hud"] += time.monotonic() - t0

        t0 = time.monotonic()
        cv2.imshow("smash-coach", composed)
        # MP4 backup recording — initialize on first frame, write each
        # composed HUD frame for a bulletproof demo fallback.
        if record_path:
            if recorder is None:
                h, w = composed.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                recorder = cv2.VideoWriter(record_path, fourcc, 30.0, (w, h))
                dlog(f"[main] recording HUD to {record_path}")
            recorder.write(composed)
        timing["draw"] += time.monotonic() - t0

        # Dashboard render (after hud) ---------------------------------
        if debug_on:
            try:
                dash_img = dashboard.render(
                    live_hud=composed,
                    raw_capture=frame.img,
                    state=s,
                    last_s1_out=last_s1,
                    last_s1_t=last_s1_t,
                    last_s2_resp=last_s2_resp,
                    s2_pending=pending_s2 is not None and not pending_s2.done(),
                    metrics=last_metrics,
                    trigger_count=trigger_count,
                    error_count=error_count,
                    now=time.monotonic(),
                    last_s2_resp_t=last_s2_resp_t,
                )
                cv2.imshow("dashboard", dash_img)
            except Exception as e:
                error_count += 1
                dlog(f"[main] dashboard render error: {e}")

        frames_since_log += 1
        if frames_since_log >= 60:
            elapsed = max(1e-6, time.monotonic() - window_start_t)
            fps = frames_since_log / elapsed
            ms = {k: 1000 * v / frames_since_log for k, v in timing.items()}
            last_metrics = {"fps": fps, **ms}
            dlog(f"[main] fps={fps:.1f}  avg ms/frame  tier0={ms['tier0']:.1f}  "
                 f"tier1={ms['tier1']:.1f}  s1_wait={ms['s1_wait']:.1f}  "
                 f"hud={ms['hud']:.1f}  draw={ms['draw']:.1f}")
            timing = {k: 0.0 for k in timing}
            frames_since_log = 0
            window_start_t = time.monotonic()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            debug_on = not debug_on
            if debug_on:
                dlog("[main] dashboard: ON")
            else:
                dlog("[main] dashboard: OFF")
                try:
                    cv2.destroyWindow("dashboard")
                except cv2.error:
                    pass

    cap.close()
    if recorder is not None:
        recorder.release()
        print(f"[main] saved recording: {record_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(run())
