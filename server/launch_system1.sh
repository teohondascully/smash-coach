#!/usr/bin/env bash
# System 1 (perception) launcher.
# Env:
#   CUDA_VISIBLE_DEVICES   GPU ids (default: 0)
#   S1_TP                  tensor parallel size (default: 1)
#   S1_PORT                port (default: 8001)
#   S1_MODEL               HF model id or local path (default: Qwen/Qwen2.5-VL-7B-Instruct)
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export S1_TP="${S1_TP:-1}"
export S1_MODEL="${S1_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${S1_PORT:-8001}"
exec uvicorn server.system1_server:app --host 0.0.0.0 --port "$PORT"
