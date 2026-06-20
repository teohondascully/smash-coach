"""Async HTTP dispatcher for System 1 (Qwen2.5-VL-7B) inference.

Sends a stack of 3 frames sampled at ~100ms intervals so the VLM has temporal
context for phase detection (startup / active / endlag distinguish much
better with motion). Crops out the UI strip before resizing — the bottom
~15% is the damage HUD which the model reads from the integer field, but
the strip is also redundant pixels to the action-recognition task.

Rate-limited to ``hz`` calls per second. On any HTTPError, returns ``None``
silently so the HUD falls through to last-known state.
"""
from __future__ import annotations

import asyncio
import base64
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import httpx
import numpy as np


@dataclass
class S1Out:
    p1: dict
    p2: dict
    t: float


# Crop: keep the play area, drop a thin top status bar and the very bottom UI.
# Damage is read by the VLM from the bottom strip, so we leave most of it in;
# this just trims the redundant edges.
_CROP_TOP_FRAC = 0.03
_CROP_BOT_FRAC = 0.05


def _prep(img: np.ndarray, size: int = 640, quality: int = 70) -> str | None:
    """Crop UI margins, resize to size×size, JPEG-encode, base64."""
    h = img.shape[0]
    top = int(h * _CROP_TOP_FRAC)
    bot = int(h * (1.0 - _CROP_BOT_FRAC))
    play = img[top:bot, :]
    small = cv2.resize(play, (size, size))
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return base64.b64encode(jpg.tobytes()).decode()


class System1Client:
    """Buffers the last ``stack_size`` frames at ~``stack_interval_s`` spacing
    and POSTs them as a single multi-image request when the rate limiter allows.
    """

    def __init__(
        self,
        url: str = "http://node:8001/infer",
        hz: float = 7.0,
        timeout: float = 4.0,
        stack_size: int = 3,
        stack_interval_s: float = 0.1,
    ):
        self.url = url
        self.min_interval = 1.0 / hz
        self.stack_size = stack_size
        self.stack_interval_s = stack_interval_s
        self._last_sent = 0.0
        self._client = httpx.AsyncClient(timeout=timeout)
        # (t, frame) pairs spaced at ~stack_interval_s.
        self._buf: deque[tuple[float, np.ndarray]] = deque(maxlen=stack_size)
        # Adaptive sampling: when both players are in "neutral", double the
        # interval each call up to 8x. Resets on the first non-neutral label.
        self._neutral_streak: int = 0
        # The S1 HTTP call runs as a background task so the capture/display loop
        # is never blocked waiting on inference (~1.2s warm). At most one request
        # is in flight; ``maybe_infer`` harvests the result when it lands.
        self._inflight: Optional[asyncio.Task] = None

    def _maybe_buffer(self, img: np.ndarray, t: float) -> None:
        if not self._buf or (t - self._buf[-1][0]) >= self.stack_interval_s:
            self._buf.append((t, img))

    async def maybe_infer(self, img: np.ndarray, t: float) -> Optional[S1Out]:
        """Non-blocking. Buffers the frame, harvests a finished inference (if
        any), and launches a new background request when the rate limiter allows
        and none is in flight. Returns a fresh ``S1Out`` exactly once per
        completed request, else ``None`` — so the caller never stalls on S1.
        """
        self._maybe_buffer(img, t)

        # Harvest a completed in-flight request (returned once, to the caller).
        result: Optional[S1Out] = None
        if self._inflight is not None and self._inflight.done():
            try:
                result = self._inflight.result()
            except Exception:  # noqa: BLE001 — network/parse errors degrade to None
                result = None
            self._inflight = None

        # Launch a new request if the limiter allows and nothing is in flight.
        now = time.monotonic()
        effective_interval = self.min_interval * (2 ** min(self._neutral_streak, 3))
        if (
            self._inflight is None
            and now - self._last_sent >= effective_interval
            and len(self._buf) >= self.stack_size
        ):
            self._last_sent = now
            frames = list(self._buf)
            self._inflight = asyncio.create_task(self._do_infer(frames, t))

        return result

    async def _do_infer(
        self, frames: list[tuple[float, np.ndarray]], t: float
    ) -> Optional[S1Out]:
        try:
            images_b64: list[str] = []
            ts: list[float] = []
            for ft, fimg in frames:
                b64 = _prep(fimg)
                if b64 is None:
                    return None
                images_b64.append(b64)
                ts.append(ft)
            r = await self._client.post(
                self.url,
                json={"images_b64": images_b64, "ts": ts},
            )
            r.raise_for_status()
            d = r.json()
            # Update the neutral streak so the next call's interval adapts.
            if d["p1"].get("action_label") == "neutral" and d["p2"].get("action_label") == "neutral":
                self._neutral_streak += 1
            else:
                self._neutral_streak = 0
            return S1Out(p1=d["p1"], p2=d["p2"], t=t)
        except (httpx.HTTPError, KeyError, ValueError):
            return None

    async def aclose(self) -> None:
        if self._inflight is not None:
            self._inflight.cancel()
        await self._client.aclose()
