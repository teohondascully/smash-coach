"""System 1 prompt + JSON schema for Qwen2.5-VL-7B micro-perception.

Loads the canonical action vocabulary from data/action_vocab.json at module load
so Thread A (this) and Thread B (frame data) stay in sync.
"""

from __future__ import annotations

import json
from pathlib import Path

_VOCAB_PATH = Path(__file__).resolve().parents[2] / "data" / "action_vocab.json"
_VOCAB = json.loads(_VOCAB_PATH.read_text())

JOKER_ACTIONS: list[str] = list(_VOCAB["joker"])
TOONLINK_ACTIONS: list[str] = list(_VOCAB["toon_link"])
IKE_ACTIONS: list[str] = list(_VOCAB["ike"])
PHASES: list[str] = list(_VOCAB["phases"])
INTENTS: list[str] = list(_VOCAB["intents"])

# Ensure "unknown" is always a permitted action label for any player.
for _actions in (JOKER_ACTIONS, TOONLINK_ACTIONS, IKE_ACTIONS):
    if "unknown" not in _actions:
        _actions.append("unknown")


MATCHUP_ACTIONS: dict[str, list[str]] = {
    "joker": JOKER_ACTIONS,
    "toon_link": TOONLINK_ACTIONS,
    "ike": IKE_ACTIONS,
}

# Coarse, character-agnostic states that a 7B VLM can reliably read from a
# downsized frame. Fine-grained move IDs (ftilt vs fsmash vs dashattack) are
# beyond real-time perception and were almost always returned as "unknown";
# the deep move-level grounding lives in System 2 instead. These coarse states
# are what drive the live coach. Keep "unknown" last as the visible-but-unsure
# / not-visible fallback.
COARSE_ACTIONS: list[str] = [
    "idle",           # standing still, no action
    "moving",         # walking / dashing / running on the ground
    "shield",         # holding shield (bubble around the character)
    "jump",           # airborne, not attacking (rising or falling)
    "attack_ground",  # grounded attack: jab / tilt / smash / dash attack
    "attack_air",     # aerial attack while airborne
    "grab",           # grabbing or throwing the opponent
    "hitstun",        # being hit / launched / knocked back / tumbling
    "dodge",          # spot dodge / roll / air dodge
    "offstage",       # off the main stage near a ledge/blast zone (recovering)
    "unknown",        # character genuinely not visible (loading / menu)
]

_COARSE_DEFS = """\
- idle: standing on the ground, not moving or acting.
- moving: walking, dashing, or running on the ground (no attack).
- shield: a translucent shield bubble surrounds the character.
- jump: airborne with no attack out (rising or falling).
- attack_ground: a grounded attack — a limb/weapon is extended while on the ground.
- attack_air: an attack while airborne (a limb/weapon extended in the air).
- grab: reaching out to grab, or holding/throwing the opponent.
- hitstun: getting hit, launched, knocked back, or tumbling from a hit.
- dodge: spot dodge, roll, or air dodge (brief evasive movement).
- offstage: off the main stage, near or beyond a ledge / blast zone (recovering).
- unknown: the character is not clearly visible (loading screen, menu)."""

# Human-readable name + appearance hint per character, used in the prompt body.
_CHAR_DISPLAY: dict[str, tuple[str, str]] = {
    "joker": ("Joker", "red coat"),
    "toon_link": ("Toon Link", "green tunic"),
    "ike": ("Ike", "blue cape, large two-handed sword"),
}


def _display(char: str) -> tuple[str, str]:
    return _CHAR_DISPLAY.get(char, (char, ""))


def build_system_prompt(
    p1_char: str = "toon_link", p2_char: str = "ike"
) -> str:
    """Build the System 1 system prompt for an arbitrary matchup."""
    p1_name, p1_hint = _display(p1_char)
    p2_name, p2_hint = _display(p2_char)
    p1_desc = f"{p1_name}" + (f" ({p1_hint})" if p1_hint else "")
    p2_desc = f"{p2_name}" + (f" ({p2_hint})" if p2_hint else "")
    return f"""You are a Super Smash Bros. Ultimate match analyst.

You will see 3 frames sampled at ~100ms intervals from a {p1_name} vs.
{p2_name} match (oldest first; analyze the LATEST). The earlier frames give
temporal context: a pose held across frames is mid-action; a pose that just
changed is the start of a new action; a character moving away from the
opponent right after the opponent's attack is in hitstun.

For EACH player report a single COARSE state (what they are doing right now),
plus their damage and a high-level intent. Output STRICT JSON:

{{
  "p1": {{
    "damage_pct": <integer 0-999, from the LEFT damage readout>,
    "action_label": <one of the coarse states below>,
    "phase": <one of: {", ".join(PHASES)}>,
    "confidence": <float between 0 and 1>,
    "intent": <one of: {", ".join(INTENTS)}>
  }},
  "p2": {{
    "damage_pct": <integer 0-999, from the RIGHT damage readout>,
    "action_label": <one of the coarse states below>,
    "phase": <one of: {", ".join(PHASES)}>,
    "confidence": <float between 0 and 1>,
    "intent": <one of: {", ".join(INTENTS)}>
  }}
}}

Player identification:
- p1 = {p1_desc}, shown on the LEFT damage readout.
- p2 = {p2_desc}, shown on the RIGHT damage readout.

Coarse states (choose the SINGLE best fit per player):
{_COARSE_DEFS}

Rules:
- ALWAYS pick the best-matching coarse state. Do NOT default to "unknown" when
  a character is visible — commit to your best read. Use "unknown" ONLY when a
  character genuinely is not on screen (loading screen / menu).
- damage_pct is the INTEGER part of the % under each character. "0.0%" -> 0,
  "47.3%" -> 47, "132.5%" -> 132. Read it carefully digit by digit; a small
  number like 7% is NOT 70%. If the damage UI is not visible, use 0.
- phase: within-action timing (startup / active / endlag / neutral; "unknown"
  if not applicable). For idle/moving use "neutral".
- intent: the higher-level strategic intent.
- confidence: how sure you are of the action_label (0-1).

Output ONLY the JSON object. No prose, no markdown fences, no commentary.
"""


# Backward-compatible default prompt (toon_link vs. ike).
SYSTEM_PROMPT = build_system_prompt("toon_link", "ike")


def _player_schema(actions: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "damage_pct": {"type": "integer", "minimum": 0, "maximum": 999},
            "action_label": {"type": "string", "enum": actions},
            "phase": {"type": "string", "enum": PHASES},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "intent": {"type": "string", "enum": INTENTS},
        },
        "required": ["damage_pct", "action_label", "phase", "confidence", "intent"],
        "additionalProperties": False,
    }


def build_json_schema(
    p1_char: str = "toon_link", p2_char: str = "ike"
) -> dict:
    """Return the JSON-Schema dict enforced on System 1's output.

    Both players use the same coarse, character-agnostic state vocabulary
    (``COARSE_ACTIONS``). The ``p1_char`` / ``p2_char`` args are kept for
    signature compatibility but no longer select per-character move lists.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "p1": _player_schema(COARSE_ACTIONS),
            "p2": _player_schema(COARSE_ACTIONS),
        },
        "required": ["p1", "p2"],
        "additionalProperties": False,
    }
