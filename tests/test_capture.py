"""Tests for mac.capture video-file source mode.

These avoid any real capture card: a tiny synthetic clip is written to a temp
file, then read back through Capture to verify file detection, frame decoding,
native-FPS pacing, and EOF looping.
"""
from __future__ import annotations

import itertools
import time

import cv2
import numpy as np
import pytest

from mac.capture import Capture, _is_file_source


def _write_clip(path: str, n_frames: int = 6, fps: float = 30.0,
                w: int = 64, h: int = 48) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "could not open VideoWriter for test clip"
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 10 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_is_file_source():
    assert _is_file_source("/tmp/gameplay.mov") is True
    assert _is_file_source("video.mp4") is True
    assert _is_file_source("0") is False  # digit-string == device index
    assert _is_file_source(0) is False
    assert _is_file_source(1) is False


def test_missing_file_raises():
    with pytest.raises(RuntimeError, match="not found"):
        Capture(device_index="/nope/does_not_exist.mp4")


def test_reads_frames_from_file(tmp_path):
    clip = str(tmp_path / "clip.mp4")
    _write_clip(clip, n_frames=6, fps=30.0)

    cap = Capture(device_index=clip)
    try:
        assert cap.is_file is True
        assert cap.src_fps == pytest.approx(30.0, abs=1.0)
        frames = list(itertools.islice(cap.frames(), 4))
        assert len(frames) == 4
        for f in frames:
            assert f.img.shape == (48, 64, 3)
            assert f.t > 0
    finally:
        cap.close()


def test_loops_past_eof(tmp_path):
    # Pull more frames than the clip contains; the generator must loop, not stop.
    clip = str(tmp_path / "clip.mp4")
    _write_clip(clip, n_frames=3, fps=60.0)

    cap = Capture(device_index=clip)
    try:
        frames = list(itertools.islice(cap.frames(), 8))
        assert len(frames) == 8  # 3 real frames, looped to reach 8
    finally:
        cap.close()


def test_exposes_frame_interval(tmp_path):
    # Pacing moved to the async caller; Capture just exposes the target interval.
    clip = str(tmp_path / "clip.mp4")
    _write_clip(clip, n_frames=10, fps=20.0)

    cap = Capture(device_index=clip)
    try:
        assert cap.frame_interval == pytest.approx(1.0 / 20.0, abs=0.01)
    finally:
        cap.close()


def test_start_sec_seeks_past_intro(tmp_path):
    # With start_sec set, the first frame should be the seeked frame, and the
    # loop point should be that frame (not 0).
    clip = str(tmp_path / "clip.mp4")
    _write_clip(clip, n_frames=20, fps=10.0)  # 2s clip

    cap = Capture(device_index=clip, start_sec=1.0)  # skip to frame ~10
    try:
        assert cap._start_frame == 10
        frames = list(itertools.islice(cap.frames(), 3))
        assert len(frames) == 3
    finally:
        cap.close()


def test_generator_does_not_block(tmp_path):
    # The file generator must yield without sleeping (caller does the pacing),
    # so pulling several frames is near-instant even for a low-FPS clip.
    clip = str(tmp_path / "clip.mp4")
    _write_clip(clip, n_frames=10, fps=5.0)  # 200ms/frame if it (wrongly) paced

    cap = Capture(device_index=clip)
    try:
        t0 = time.monotonic()
        list(itertools.islice(cap.frames(), 8))
        elapsed = time.monotonic() - t0
        assert elapsed < 0.2  # would be ~1.4s if it still slept per frame
    finally:
        cap.close()
