"""System 2 client: keyframe selection + async HTTP dispatch.

Implements Task 5.3 of the smash-coach plan.
"""
from __future__ import annotations

import base64
from typing import Optional

import cv2
import httpx
import numpy as np

from mac.state import StateT
from mac.trigger import TriggerEvent


def select_keyframes(
    buf: list[tuple[float, np.ndarray]],
    trajectory: list[StateT],
    max_n: int = 8,
) -> list[tuple[float, np.ndarray]]:
    """Pick frames at action onsets in the trajectory plus the last frame.

    For each trajectory state where any player's action label is not "neutral"
    or "walk", we take its timestamp as a target. We also always include the
    most recent trajectory timestamp. For each target time, we find the nearest
    buffer entry by absolute time delta. We then keep the ``max_n`` newest
    selections (de-duplicated by buffer index, ordered by timestamp ascending).
    """
    if not buf or not trajectory:
        return []

    target_times: list[float] = []
    for s in trajectory:
        actions = s.actions or {}
        for who, a in actions.items():
            if a.label not in {"neutral", "walk"}:
                target_times.append(s.t)
                break

    # Always include the last trajectory frame.
    target_times.append(trajectory[-1].t)

    # Map targets to nearest buffer index.
    buf_times = [bt for bt, _ in buf]
    chosen_indices: list[int] = []
    for t in target_times:
        best_i = min(range(len(buf)), key=lambda i: abs(buf_times[i] - t))
        if best_i not in chosen_indices:
            chosen_indices.append(best_i)

    # Keep the newest ``max_n`` by buffer-time, then sort ascending.
    chosen_indices.sort(key=lambda i: buf_times[i])
    if len(chosen_indices) > max_n:
        chosen_indices = chosen_indices[-max_n:]

    return [buf[i] for i in chosen_indices]


class System2Client:
    """Async HTTP client that posts a trigger payload to the System 2 server."""

    def __init__(self, url: str):
        self.url = url
        self._client = httpx.AsyncClient(timeout=20.0)

    async def request(
        self,
        event: TriggerEvent,
        trajectory: list[StateT],
        keyframes: list[tuple[float, np.ndarray]],
    ) -> Optional[dict]:
        encoded: list[dict] = []
        for t, frame in keyframes:
            resized = cv2.resize(frame, (640, 640))
            ok, jpg = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
            encoded.append({"image_b64": b64, "t": float(t)})

        body = {
            "state_trajectory": [s.model_dump() for s in trajectory],
            "keyframes": encoded,
            "event_type": event.kind,
        }

        try:
            r = await self._client.post(self.url, json=body)
            r.raise_for_status()
        except httpx.HTTPError:
            return None
        return r.json()

    async def aclose(self) -> None:
        await self._client.aclose()
