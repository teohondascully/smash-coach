"""Tests for the precomputed label track (playback lookup + damage cleaning)."""
from __future__ import annotations

from mac.label_track import LabelTrack, _median3


def _entry(t, d1, d2, a1="idle", a2="idle"):
    return {"video_t": t,
            "p1": {"action_label": a1, "damage_pct": d1, "phase": "neutral",
                   "confidence": 0.9, "intent": "neutral"},
            "p2": {"action_label": a2, "damage_pct": d2, "phase": "neutral",
                   "confidence": 0.9, "intent": "neutral"}}


def test_median3_removes_isolated_outlier():
    assert _median3([51, 6, 61]) == [51, 51, 61]
    assert _median3([0, 0, 0]) == [0, 0, 0]


def test_median3_preserves_sustained_change():
    # a real stock reset (sustained 0s) must survive
    assert _median3([88, 0, 0, 0]) == [88, 0, 0, 0]


def test_lookup_holds_last_state():
    tr = LabelTrack([_entry(11.0, 0, 0), _entry(11.5, 0, 10), _entry(12.0, 0, 20)])
    assert tr.at(11.7).p2["damage_pct"] == 10   # most recent <= 11.7
    assert tr.at(99.0).p2["damage_pct"] == 20   # past end -> last
    assert tr.at(0.0).p2["damage_pct"] == 0     # before start -> first


def test_damage_cleaned_on_load():
    tr = LabelTrack([_entry(1, 51, 0), _entry(2, 6, 0), _entry(3, 61, 0)])
    assert tr.at(2.0).p1["damage_pct"] == 51    # the stray 6 is scrubbed


def test_unsorted_entries_are_sorted():
    tr = LabelTrack([_entry(3.0, 0, 30), _entry(1.0, 0, 10), _entry(2.0, 0, 20)])
    assert tr._ts == [1.0, 2.0, 3.0]
