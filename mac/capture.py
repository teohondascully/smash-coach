"""UVC capture loop for Elgato HD60 X (or any AVFoundation device on macOS)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator

import cv2


@dataclass
class Frame:
    img: Any  # np.ndarray BGR
    t: float  # monotonic timestamp (seconds)


class Capture:
    """Thin wrapper around cv2.VideoCapture targeting an AVFoundation UVC device.

    Per the implementation plan, the constructor raises ``RuntimeError`` if the
    capture device cannot be opened. Smoke scripts catch this so the code path is
    still runnable in environments without a capture card attached.
    """

    def __init__(self, device_index: int = 0, width: int = 1920, height: int = 1080):
        self.device_index = device_index
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        # Force MJPG over raw YUV — needs ~10x less USB bandwidth, prevents
        # silent fps downgrade on hubs / shared USB controllers.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Smallest possible buffer so .read() always returns the most recent
        # frame rather than a queued stale one when processing falls behind.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Capture device {device_index} not opened "
                f"(check that the capture card is plugged in and granted permission)."
            )

    def frames(self) -> Iterator[Frame]:
        while True:
            ok, img = self.cap.read()
            if not ok:
                # transient read failure — keep trying
                continue
            yield Frame(img=img, t=time.monotonic())

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
