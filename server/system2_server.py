"""System 2 counterfactual server (FastAPI + SGLang Qwen2.5-VL-72B-AWQ).

Heavy GPU imports (`sglang`) are performed lazily inside the startup handler so
this module is importable on a Mac dev box that has no sglang installed.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from server.prompts.system2 import COUNTERFACTUAL_SCHEMA, build_system_prompt

logger = logging.getLogger("system2")

app = FastAPI(title="Smash Coach System 2")

_runtime: Any = None
_sgl: Any = None
_analyze_fn: Any = None
SYS_PROMPT: str | None = None

_FRAME_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "frame_data.json"
)


class Keyframe(BaseModel):
    image_b64: str
    t: float


class CounterfactualReq(BaseModel):
    state_trajectory: list[dict]
    keyframes: list[Keyframe]
    event_type: str  # "stock_loss" | "exchange"


def _load_frame_data_blob() -> str:
    if _FRAME_DATA_PATH.exists():
        return _FRAME_DATA_PATH.read_text()
    logger.warning(
        "data/frame_data.json not found at %s — using empty frame-data blob. "
        "Thread B should produce this before live use.",
        _FRAME_DATA_PATH,
    )
    return ""


@app.on_event("startup")
def load_model() -> None:
    """Load Qwen2.5-VL-72B-AWQ via SGLang. Lazy-imports sglang here."""
    global _runtime, _sgl, _analyze_fn, SYS_PROMPT

    # Lazy import — sglang is GPU-only and not installed on macOS.
    import sglang as sgl  # type: ignore

    _sgl = sgl

    logger.info("Loading Qwen2.5-VL-72B-Instruct-AWQ on TP=4...")
    _runtime = sgl.Runtime(
        model_path="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        tp_size=4,
        mem_fraction_static=0.85,
    )
    sgl.set_default_backend(_runtime)

    SYS_PROMPT = build_system_prompt(_load_frame_data_blob())

    @sgl.function
    def analyze(s, system_prompt, user_text, images):  # type: ignore
        s += sgl.system(system_prompt)
        if images:
            s += sgl.user(sgl.image(images[0]))
            for img in images[1:]:
                s += sgl.user(sgl.image(img))
        s += sgl.user(user_text)
        s += sgl.assistant(
            sgl.gen(
                "out",
                max_tokens=800,
                temperature=0.2,
                json_schema=json.dumps(COUNTERFACTUAL_SCHEMA),
            )
        )

    _analyze_fn = analyze
    logger.info("System 2 ready.")


@app.get("/health")
def health() -> dict:
    return {"ok": _runtime is not None}


@app.post("/counterfactual")
def counterfactual(req: CounterfactualReq) -> dict:
    if _analyze_fn is None or SYS_PROMPT is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        images = [
            Image.open(io.BytesIO(base64.b64decode(kf.image_b64))).convert("RGB")
            for kf in req.keyframes
        ]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad image: {e}") from e

    user_text = (
        f"Event: {req.event_type}\n"
        f"State trajectory ({len(req.state_trajectory)} steps):\n"
        f"{json.dumps(req.state_trajectory, indent=2)}\n\n"
        "Analyze this exchange and produce the counterfactual JSON."
    )

    state = _analyze_fn.run(
        system_prompt=SYS_PROMPT,
        user_text=user_text,
        images=images,
    )
    text = state["out"]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("model returned non-JSON: %s", text[:200])
        raise HTTPException(
            status_code=500, detail=f"model returned non-JSON: {e}"
        ) from e
