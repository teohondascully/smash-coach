"""System 2 prompt + JSON schema for Qwen2.5-VL-72B counterfactual analysis."""

from __future__ import annotations


COUNTERFACTUAL_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": 400},
        "chosen_action": {
            "type": "object",
            "properties": {
                "player": {"type": "string", "enum": ["p1", "p2"]},
                "action_label": {"type": "string"},
                "frame_t": {"type": "number"},
                "reasoning": {"type": "string", "maxLength": 300},
            },
            "required": ["player", "action_label", "frame_t", "reasoning"],
            "additionalProperties": False,
        },
        "counterfactual_action": {
            "type": "object",
            "properties": {
                "action_label": {"type": "string"},
                "rationale": {"type": "string", "maxLength": 300},
                "qualitative_likelihood": {
                    "type": "string",
                    "enum": ["likely", "plausible", "speculative"],
                },
            },
            "required": [
                "action_label",
                "rationale",
                "qualitative_likelihood",
            ],
            "additionalProperties": False,
        },
        "frame_data_citations": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "character": {"type": "string"},
                    "move": {"type": "string"},
                    "stat": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["character", "move", "stat", "value"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "summary",
        "chosen_action",
        "counterfactual_action",
        "frame_data_citations",
    ],
    "additionalProperties": False,
}


_PROMPT_TEMPLATE = """You are a Super Smash Bros. Ultimate frame-data analyst.
You analyze a brief (~5 second) exchange between Joker (p1) and Toon Link (p2)
and produce a single counterfactual: what actually happened vs. one grounded
alternative.

You will be given:
  1. The authoritative FRAME DATA TABLE for both characters (inlined below).
  2. A timestamped state trajectory (a list of s_t dicts).
  3. A small set of saliency keyframes (images sampled at action onsets).

Output STRICT JSON matching the provided schema. Rules:
  - chosen_action.action_label MUST be a move that appears in the frame data
    table. counterfactual_action.action_label MUST also appear there.
  - Cite real moves and real numbers only. Never invent move names, frame
    counts, or character stats. If you are not sure, omit the citation.
  - You MAY include up to 6 entries in frame_data_citations.
  - For likelihood, use ONLY the qualitative bands "likely", "plausible",
    or "speculative". NEVER output a numerical probability, percentage,
    or odds. NEVER write things like "70%", "p=0.4", or "70 percent".
    Qualitative bands only.
  - "summary" must be <= 400 chars; reasoning/rationale <= 300 chars each.
  - Output ONLY the JSON object. No prose, no markdown fences.

FRAME DATA TABLE:
{frame_data_blob}
"""


def build_system_prompt(frame_data_blob: str) -> str:
    """Return the System 2 prompt with the frame-data table inlined.

    The caller passes the raw frame_data.json text (or a pre-formatted summary).
    """
    return _PROMPT_TEMPLATE.format(frame_data_blob=frame_data_blob)
