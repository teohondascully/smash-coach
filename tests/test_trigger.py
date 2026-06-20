"""Tests for mac.trigger."""
from __future__ import annotations

from mac.state import ActionState, PlayerState, StateT
from mac.trigger import TriggerDetector


def make_state(t: float) -> StateT:
    return StateT(
        t=t,
        damage={"p1": 0.0, "p2": 0.0},
        stocks={"p1": 1, "p2": 1},
        positions={
            "p1": PlayerState(x=0.0, y=0.0, facing="right", airborne=False),
            "p2": PlayerState(x=100.0, y=0.0, facing="left", airborne=False),
        },
        actions={
            "p1": ActionState(label="neutral"),
            "p2": ActionState(label="neutral"),
        },
    )


def test_stock_loss_fires():
    td = TriggerDetector()
    s1 = make_state(0.0)
    s1.stocks = {"p1": 1, "p2": 1}
    s2 = make_state(0.5)
    s2.stocks = {"p1": 0, "p2": 1}
    assert td.check(s1) is None
    ev = td.check(s2)
    assert ev is not None
    assert ev.kind == "stock_loss"
    assert ev.who == "p1"


def test_damage_spike_fires():
    td = TriggerDetector(damage_delta=30.0, damage_window_s=2.0)
    s1 = make_state(0.0)
    s1.damage = {"p1": 0.0, "p2": 0.0}
    s2 = make_state(1.0)
    s2.damage = {"p1": 35.0, "p2": 0.0}
    td.check(s1)
    ev = td.check(s2)
    assert ev is not None
    assert ev.kind == "exchange"
    assert ev.who == "p1"


def test_cooldown_prevents_double_fire():
    td = TriggerDetector(damage_delta=30.0, cooldown_s=5.0)
    s1 = make_state(0.0)
    s1.damage = {"p1": 0.0, "p2": 0.0}
    s2 = make_state(1.0)
    s2.damage = {"p1": 35.0, "p2": 0.0}
    s3 = make_state(1.5)
    s3.damage = {"p1": 70.0, "p2": 0.0}
    td.check(s1)
    ev1 = td.check(s2)
    ev2 = td.check(s3)
    assert ev1 is not None
    assert ev2 is None
