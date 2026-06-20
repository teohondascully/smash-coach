"""Tests for mac.scorer."""
from __future__ import annotations

from mac.frame_data import FrameData
from mac.scorer import score_counterfactual

FD_PATH = "data/frame_data.json"
HB_PATH = "data/hitboxes.json"


def _fd() -> FrameData:
    return FrameData.load(FD_PATH, HB_PATH)


def test_score_spotdodge_punish_window():
    fd = _fd()
    out = score_counterfactual(
        fd,
        attacker_char="toon_link",
        attacker_move="fair",
        defender_char="joker",
        defender_response="spotdodge",
    )
    assert "punish_window_frames" in out
    assert isinstance(out["punish_window_frames"], int)
    assert out["punish_window_frames"] >= 0


def test_score_shield_frame_advantage():
    fd = _fd()
    out = score_counterfactual(
        fd,
        attacker_char="joker",
        attacker_move="fsmash",
        defender_char="toon_link",
        defender_response="shield",
    )
    # fsmash on shield should give the defender positive frame advantage.
    assert out["frame_advantage"] is not None
    assert out["frame_advantage"] > 0
    assert out["punish_window_frames"] >= 0


def test_score_unknown_move_does_not_crash():
    fd = _fd()
    out = score_counterfactual(
        fd,
        attacker_char="joker",
        attacker_move="does_not_exist",
        defender_char="toon_link",
        defender_response="shield",
    )
    assert out["punish_window_frames"] == 0
    assert any("no frame data" in n for n in out["notes"])
