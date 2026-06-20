"""UVC capture loop for Elgato HD60 X (or any AVFoundation device on macOS).

Also supports a *video-file* source: if constructed with a path to a video
(e.g. ``/path/gameplay.mov``) instead of a device index, frames are read from
the file, paced to the file's native FPS, and looped on EOF. This lets the
whole pipeline run off a recorded clip on a machine with no capture card.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Iterator

import cv2


@dataclass
class Frame:
    img: Any  # np.ndarray BGR
    t: float  # monotonic timestamp (seconds)


def _is_file_source(source: Any) -> bool:
    """A source is a file path if it's a non-digit string (e.g. a video path)."""
    return isinstance(source, str) and not source.isdigit()


class Capture:
    """Thin wrapper around cv2.VideoCapture.

    ``device_index`` accepts either:
      - an int (or digit-string) → opens that AVFoundation UVC device (capture card)
      - a path string (e.g. ``/path/gameplay.mov``) → opens the video file,
        paced to its native FPS and looped on EOF.

    The constructor raises ``RuntimeError`` if the source cannot be opened. Smoke
    scripts catch this so the code path is still runnable without a capture card.
    """

    def __init__(self, device_index: Any = 0, width: int = 1920, height: int = 1080,
                 start_sec: float = 0.0):
        self.device_index = device_index
        self.is_file = _is_file_source(device_index)
        self._start_frame = 0

        if self.is_file:
            if not os.path.exists(device_index):
                raise RuntimeError(f"Capture file {device_index!r} not found.")
            self.cap = cv2.VideoCapture(device_index)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Capture file {device_index!r} could not be opened "
                    f"(unsupported codec? try re-encoding to H.264)."
                )
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            # Fall back to 30fps if the container doesn't report a sane FPS.
            self.src_fps = fps if fps and fps > 0 else 30.0
            # Target seconds-per-frame. Pacing is done by the (async) caller via
            # asyncio.sleep so background inference tasks aren't starved; the
            # generator itself never blocks. 0.0 for live devices (no pacing).
            self.frame_interval = 1.0 / self.src_fps
            # Skip an intro (menus/character-select) and loop back to it, not 0.
            if start_sec > 0:
                self._start_frame = int(start_sec * self.src_fps)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)
        else:
            idx = int(device_index)
            self.cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            # Force MJPG over raw YUV — needs ~10x less USB bandwidth, prevents
            # silent fps downgrade on hubs / shared USB controllers.
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            # Smallest possible buffer so .read() always returns the most recent
            # frame rather than a queued stale one when processing falls behind.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.frame_interval = 0.0
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Capture device {device_index} not opened "
                    f"(check that the capture card is plugged in and granted permission)."
                )

    def frames(self) -> Iterator[Frame]:
        if self.is_file:
            yield from self._file_frames()
        else:
            yield from self._device_frames()

    def _device_frames(self) -> Iterator[Frame]:
        while True:
            ok, img = self.cap.read()
            if not ok:
                # transient read failure — keep trying
                continue
            yield Frame(img=img, t=time.monotonic())

    def _file_frames(self) -> Iterator[Frame]:
        # Yield frames as fast as requested; the async caller paces playback to
        # ``frame_interval`` with asyncio.sleep (which also lets background
        # inference tasks run). Loops back to the start on EOF.
        while True:
            ok, img = self.cap.read()
            if not ok:
                # EOF → loop back to the start frame (past any skipped intro).
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)
                ok, img = self.cap.read()
                if not ok:
                    return  # genuinely unreadable; stop cleanly
            yield Frame(img=img, t=time.monotonic())

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
