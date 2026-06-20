"""Schema and lazy-import smoke tests for the System 1 / System 2 prompts.

These tests are the primary verification surface on Mac (no GPU available).
"""

from __future__ import annotations

import jsonschema
import pytest


# ---------------------------------------------------------------------------
# System 1 schema
# ---------------------------------------------------------------------------


def test_system1_schema_is_valid_jsonschema():
    from server.prompts.system1 import build_json_schema

    schema = build_json_schema()
    # Should not raise.
    jsonschema.Draft7Validator.check_schema(schema)


def test_system1_schema_accepts_valid_sample():
    from server.prompts.system1 import (
        JOKER_ACTIONS,
        TOONLINK_ACTIONS,
        build_json_schema,
    )

    sample = {
        "p1": {
            "action_label": JOKER_ACTIONS[0],
            "phase": "neutral",
            "confidence": 0.5,
            "intent": "neutral",
        },
        "p2": {
            "action_label": TOONLINK_ACTIONS[0],
            "phase": "startup",
            "confidence": 0.9,
            "intent": "pressuring",
        },
    }
    jsonschema.validate(instance=sample, schema=build_json_schema())


def test_system1_schema_rejects_invalid_action_label():
    from server.prompts.system1 import build_json_schema

    bad = {
        "p1": {
            "action_label": "definitely_not_a_real_joker_move",
            "phase": "neutral",
            "confidence": 0.5,
            "intent": "neutral",
        },
        "p2": {
            "action_label": "boomerang",
            "phase": "active",
            "confidence": 0.8,
            "intent": "pressuring",
        },
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=build_json_schema())


def test_system1_system_prompt_mentions_json():
    from server.prompts.system1 import SYSTEM_PROMPT

    assert "JSON" in SYSTEM_PROMPT or "json" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# System 2 schema
# ---------------------------------------------------------------------------


def test_system2_schema_is_valid_jsonschema():
    from server.prompts.system2 import COUNTERFACTUAL_SCHEMA

    jsonschema.Draft7Validator.check_schema(COUNTERFACTUAL_SCHEMA)


def test_system2_schema_accepts_valid_sample():
    from server.prompts.system2 import COUNTERFACTUAL_SCHEMA

    sample = {
        "summary": "Joker shielded the Toon Link fsmash but rolled behind, eating a punish.",
        "chosen_action": {
            "player": "p1",
            "action_label": "roll",
            "frame_t": 1.2,
            "reasoning": "Player rolled away from shield pressure, predictable.",
        },
        "counterfactual_action": {
            "action_label": "spotdodge",
            "rationale": "Spotdodge keeps stage position and opens a punish window.",
            "qualitative_likelihood": "likely",
        },
        "frame_data_citations": [
            {
                "character": "toon_link",
                "move": "fsmash",
                "stat": "startup_f",
                "value": "15",
            }
        ],
    }
    jsonschema.validate(instance=sample, schema=COUNTERFACTUAL_SCHEMA)


def test_system2_schema_rejects_numeric_likelihood():
    from server.prompts.system2 import COUNTERFACTUAL_SCHEMA

    bad = {
        "summary": "x",
        "chosen_action": {
            "player": "p1",
            "action_label": "roll",
            "frame_t": 0.0,
            "reasoning": "x",
        },
        "counterfactual_action": {
            "action_label": "spotdodge",
            "rationale": "x",
            "qualitative_likelihood": "0.87",
        },
        "frame_data_citations": [],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=COUNTERFACTUAL_SCHEMA)


def test_system2_build_system_prompt_inlines_blob():
    from server.prompts.system2 import build_system_prompt

    prompt = build_system_prompt("FOO_FRAME_DATA")
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "FOO_FRAME_DATA" in prompt


# ---------------------------------------------------------------------------
# Lazy-import discipline: the FastAPI modules must import on Mac
# without vllm/sglang installed.
# ---------------------------------------------------------------------------


def test_system1_server_module_importable():
    try:
        import server.system1_server  # noqa: F401
    except ImportError as e:  # pragma: no cover - explicit fail message
        pytest.fail(f"server.system1_server failed to import: {e}")


def test_system2_server_module_importable():
    try:
        import server.system2_server  # noqa: F401
    except ImportError as e:  # pragma: no cover
        pytest.fail(f"server.system2_server failed to import: {e}")
