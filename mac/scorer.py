"""Deterministic rule-based scorer for counterfactual analysis.

Given an attacker move and a defender response, compute punish window /
frame advantage from frame data. These numbers are what the rewind card
displays; the VLM only supplies qualitative likelihood.
"""
from __future__ import annotations

from typing import Any

from mac.frame_data import FrameData


# Approximate fastest follow-up frames for each character.
# OoS = out-of-shield (e.g. jumpsquat + aerial, or up-B).
# ground_dodge_to_attack = frames from dodge end to first available attack.
FASTEST_PUNISH: dict[str, dict[str, int]] = {
    "joker": {"OoS": 6, "ground_dodge_to_attack": 4},
    "toon_link": {"OoS": 6, "ground_dodge_to_attack": 5},
}


def score_counterfactual(
    fd: FrameData,
    attacker_char: str,
    attacker_move: str,
    defender_char: str,
    defender_response: str,
) -> dict[str, Any]:
    """Return deterministic frame-counts for the rewind card.

    The output dict always contains:
      - punish_window_frames: int (>=0)
      - frame_advantage: Optional[int] (defender perspective; None unless shield)
      - notes: list[str]
    """
    out: dict[str, Any] = {
        "punish_window_frames": 0,
        "frame_advantage": None,
        "notes": [],
    }
    try:
        m = fd.move(attacker_char, attacker_move)
    except KeyError:
        out["notes"].append(f"no frame data for {attacker_char}.{attacker_move}")
        return out

    punish_tbl = FASTEST_PUNISH.get(defender_char, {"OoS": 6, "ground_dodge_to_attack": 5})

    if defender_response == "shield" and m.shield_advantage is not None:
        adv_for_defender = -m.shield_advantage
        out["frame_advantage"] = adv_for_defender
        if adv_for_defender > 0:
            out["punish_window_frames"] = max(0, adv_for_defender - punish_tbl["OoS"])
        else:
            out["notes"].append("shielding does not yield a punish window")
    elif defender_response in ("spotdodge", "roll", "airdodge"):
        if m.active_f is not None and m.endlag_f is not None:
            recovery_after_active = m.endlag_f - m.active_f[1]
            out["punish_window_frames"] = max(
                0, recovery_after_active - punish_tbl["ground_dodge_to_attack"]
            )
        else:
            out["notes"].append("missing active/endlag frame data")
    else:
        out["notes"].append(
            f"no scoring rule for defender_response={defender_response}"
        )
    return out
