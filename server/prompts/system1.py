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
    p1_actions = MATCHUP_ACTIONS[p1_char]
    p2_actions = MATCHUP_ACTIONS[p2_char]
    p1_name, p1_hint = _display(p1_char)
    p2_name, p2_hint = _display(p2_char)
    p1_desc = f"{p1_name}" + (f" ({p1_hint})" if p1_hint else "")
    p2_desc = f"{p2_name}" + (f" ({p2_hint})" if p2_hint else "")
    return f"""You are a Super Smash Bros. Ultimate frame analyst.

You will see 3 frames sampled at ~100ms intervals from a {p1_name} vs.
{p2_name} match (oldest first; analyze the LATEST). The earlier frames
give you temporal context to disambiguate move phases — a move with the
same pose across 2 frames is mid-active; transitions between distinct
poses are startup or endlag. Read
the damage % readouts at the bottom of the screen and identify what each
player is doing. Output STRICT JSON:

{{
  "p1": {{
    "damage_pct": <integer 0-999, parsed from the LEFT damage readout>,
    "action_label": <one of the {p1_name.upper()} actions below>,
    "phase": <one of: {", ".join(PHASES)}>,
    "confidence": <float between 0 and 1>,
    "intent": <one of: {", ".join(INTENTS)}>
  }},
  "p2": {{
    "damage_pct": <integer 0-999, parsed from the RIGHT damage readout>,
    "action_label": <one of the {p2_name.upper()} actions below>,
    "phase": <one of: {", ".join(PHASES)}>,
    "confidence": <float between 0 and 1>,
    "intent": <one of: {", ".join(INTENTS)}>
  }}
}}

Conventions:
- p1 = {p1_desc} (left side at match start). p2 = {p2_desc}.
- damage_pct is the INTEGER part of the percentage shown under each
  character portrait. "0.0%" -> 0. "47.3%" -> 47. "100.0%" -> 100.
  If you can't see the damage UI (loading screen, between matches), use 0.
- If you cannot identify the action, use "unknown" with confidence < 0.3.
- "phase" is the within-move phase: startup, active, endlag, or neutral
  (use "unknown" when uncertain).
- "intent" is the higher-level strategic intent.

Allowed {p1_name.upper()} action_label values:
{", ".join(p1_actions)}

Allowed {p2_name.upper()} action_label values:
{", ".join(p2_actions)}

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
    """Return the JSON-Schema dict enforced on System 1's output."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "p1": _player_schema(MATCHUP_ACTIONS[p1_char]),
            "p2": _player_schema(MATCHUP_ACTIONS[p2_char]),
        },
        "required": ["p1", "p2"],
        "additionalProperties": False,
    }
