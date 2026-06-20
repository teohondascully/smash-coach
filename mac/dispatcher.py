"""Async HTTP dispatcher for System 1 (Qwen2.5-VL-7B) inference.

Rate-limited to ``hz`` calls per second. Downsamples to 640x640 JPEG quality 70
to keep upstream bandwidth low. On any HTTPError / KeyError, returns ``None``
silently — the HUD falls through to last-known state.
"""
from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import httpx


@dataclass
class S1Out:
    p1: dict
    p2: dict
    t: float


class System1Client:
    def __init__(
        self,
        url: str = "http://node:8001/infer",
        hz: float = 7.0,
        timeout: float = 2.0,
    ):
        self.url = url
        self.min_interval = 1.0 / hz
        self._last_sent = 0.0
        self._client = httpx.AsyncClient(timeout=timeout)

    async def maybe_infer(self, img, t: float) -> Optional[S1Out]:
        now = time.monotonic()
        if now - self._last_sent < self.min_interval:
            return None
        self._last_sent = now
        try:
            small = cv2.resize(img, (640, 640))
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                return None
            b64 = base64.b64encode(jpg.tobytes()).decode()
            r = await self._client.post(
                self.url, json={"image_b64": b64, "t": t}
            )
            r.raise_for_status()
            d = r.json()
            return S1Out(p1=d["p1"], p2=d["p2"], t=t)
        except (httpx.HTTPError, KeyError, ValueError):
            return None

    async def aclose(self) -> None:
        await self._client.aclose()
