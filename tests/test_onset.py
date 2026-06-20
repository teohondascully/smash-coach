"""Tests for mac.onset."""
from __future__ import annotations

from mac.onset import OnsetTracker


def test_onset_records_transition():
    tr = OnsetTracker()
    tr.update("p1", "neutral", 1.0)
    tr.update("p1", "neutral", 1.1)
    tr.update("p1", "fsmash", 1.2)
    assert tr.onset("p1") == 1.2


def test_onset_persists_during_action():
    tr = OnsetTracker()
    tr.update("p1", "fsmash", 1.0)
    tr.update("p1", "fsmash", 1.2)
    assert tr.onset("p1") == 1.0


def test_onset_resets_on_new_action():
    tr = OnsetTracker()
    tr.update("p1", "fsmash", 1.0)
    tr.update("p1", "shield", 1.5)
    assert tr.onset("p1") == 1.5


def test_frame_in_move():
    tr = OnsetTracker()
    tr.update("p1", "fsmash", 1.0)
    # 0.25s later -> 15 frames at 60Hz
    assert tr.frame_in_move("p1", 1.25) == 15
    # Unknown player
    assert tr.frame_in_move("p2", 1.0) == 0
