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
PHASES: list[str] = list(_VOCAB["phases"])
INTENTS: list[str] = list(_VOCAB["intents"])

# Ensure "unknown" is always a permitted action label for either player.
if "unknown" not in JOKER_ACTIONS:
    JOKER_ACTIONS.append("unknown")
if "unknown" not in TOONLINK_ACTIONS:
    TOONLINK_ACTIONS.append("unknown")


SYSTEM_PROMPT = f"""You are a Super Smash Bros. Ultimate frame analyst.

You will see a single game frame from a Joker vs. Toon Link match on Final
Destination. Identify what each player is doing on this frame and output
STRICT JSON with the following shape:

{{
  "p1": {{
    "action_label": <one of the JOKER actions below>,
    "phase": <one of: {", ".join(PHASES)}>,
    "confidence": <float between 0 and 1>,
    "intent": <one of: {", ".join(INTENTS)}>
  }},
  "p2": {{
    "action_label": <one of the TOON LINK actions below>,
    "phase": <one of: {", ".join(PHASES)}>,
    "confidence": <float between 0 and 1>,
    "intent": <one of: {", ".join(INTENTS)}>
  }}
}}

Conventions:
- p1 = Joker (red coat). p2 = Toon Link (green tunic).
- If you cannot identify the action, use "unknown" with confidence < 0.3.
- "phase" is the within-move phase: startup, active, endlag, or neutral
  (use "unknown" when uncertain).
- "intent" is the higher-level strategic intent.

Allowed JOKER action_label values:
{", ".join(JOKER_ACTIONS)}

Allowed TOON LINK action_label values:
{", ".join(TOONLINK_ACTIONS)}

Output ONLY the JSON object. No prose, no markdown fences, no commentary.
"""


def _player_schema(actions: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "action_label": {"type": "string", "enum": actions},
            "phase": {"type": "string", "enum": PHASES},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "intent": {"type": "string", "enum": INTENTS},
        },
        "required": ["action_label", "phase", "confidence", "intent"],
        "additionalProperties": False,
    }


def build_json_schema() -> dict:
    """Return the JSON-Schema dict enforced on System 1's output."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "p1": _player_schema(JOKER_ACTIONS),
            "p2": _player_schema(TOONLINK_ACTIONS),
        },
        "required": ["p1", "p2"],
        "additionalProperties": False,
    }
