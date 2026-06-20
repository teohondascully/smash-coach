"""Tests for mac.frame_data."""
from __future__ import annotations

from mac.frame_data import FrameData, HitboxCircle, Move

FD_PATH = "data/frame_data.json"
HB_PATH = "data/hitboxes.json"


def test_load_succeeds():
    fd = FrameData.load(FD_PATH, HB_PATH)
    assert "joker" in fd.moves
    assert "toon_link" in fd.moves
    assert len(fd.moves["joker"]) >= 14
    assert len(fd.moves["toon_link"]) >= 14


def test_lookup_known_move():
    fd = FrameData.load(FD_PATH, HB_PATH)
    m = fd.move("joker", "fsmash")
    assert isinstance(m, Move)
    assert m.startup_f is not None and m.startup_f > 0


def test_hitbox_active_window():
    fd = FrameData.load(FD_PATH, HB_PATH)
    circles = fd.hitboxes("joker", "fsmash", 17)
    assert len(circles) >= 1
    assert all(isinstance(c, HitboxCircle) for c in circles)


def test_hitbox_outside_window_empty():
    fd = FrameData.load(FD_PATH, HB_PATH)
    assert fd.hitboxes("joker", "fsmash", 99) == []
    assert fd.hitboxes("toon_link", "fsmash", 99) == []


def test_load_missing_hitboxes_does_not_crash(tmp_path):
    fd = FrameData.load(FD_PATH, str(tmp_path / "does_not_exist.json"))
    assert fd.hitboxes_data == {}
    assert fd.hitboxes("joker", "fsmash", 17) == []
