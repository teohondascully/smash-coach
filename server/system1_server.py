"""System 1 micro-perception server (FastAPI + vLLM Qwen2.5-VL-7B).

Heavy GPU imports (`vllm`) are performed lazily inside the startup handler so
this module is importable on a Mac dev box that has no vllm installed.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from server.prompts.system1 import build_json_schema, build_system_prompt

logger = logging.getLogger("system1")

app = FastAPI(title="Smash Coach System 1")

# Module-level singletons populated at startup.
_llm: Any = None
_SamplingParams: Any = None
_GuidedDecodingParams: Any = None

P1_CHAR = os.getenv("P1_CHAR", "toon_link")
P2_CHAR = os.getenv("P2_CHAR", "ike")
SCHEMA = build_json_schema(P1_CHAR, P2_CHAR)
SYSTEM_PROMPT = build_system_prompt(P1_CHAR, P2_CHAR)


class FrameIn(BaseModel):
    image_b64: str  # base64-encoded JPEG (or PNG) bytes
    t: float


@app.on_event("startup")
def load_model() -> None:
    """Load Qwen2.5-VL-7B via vLLM. Imports vLLM lazily here."""
    global _llm, _SamplingParams, _GuidedDecodingParams

    # Lazy import: vllm is GPU-only and not installed on macOS.
    from vllm import LLM, SamplingParams  # type: ignore
    from vllm.sampling_params import GuidedDecodingParams  # type: ignore

    _SamplingParams = SamplingParams
    _GuidedDecodingParams = GuidedDecodingParams

    logger.info("Loading Qwen2.5-VL-7B-Instruct on TP=2...")
    _llm = LLM(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
    )
    logger.info("System 1 ready.")


@app.get("/health")
def health() -> dict:
    return {"ok": _llm is not None}


@app.post("/infer")
def infer(req: FrameIn) -> dict:
    if _llm is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        img_bytes = base64.b64decode(req.image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad image: {e}") from e

    sampling = _SamplingParams(
        temperature=0.0,
        max_tokens=256,
        guided_decoding=_GuidedDecodingParams(json=SCHEMA),
    )
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"Analyze this frame at t={req.t:.3f}s."},
            ],
        },
    ]
    out = _llm.chat(chat, sampling_params=sampling)
    text = out[0].outputs[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Grammar should prevent this, but degrade gracefully.
        logger.warning("model returned non-JSON: %s", text[:200])
        raise HTTPException(
            status_code=500, detail=f"model returned non-JSON: {e}"
        ) from e
