#!/usr/bin/env bash
# System 2 (counterfactual) launcher.
# Env:
#   CUDA_VISIBLE_DEVICES   GPU ids (default: 1)
#   S2_TP                  tensor parallel size (default: 1)
#   S2_PORT                port (default: 8002)
#   S2_MODEL               HF model id or local path (default: Qwen/Qwen2.5-VL-72B-Instruct-AWQ)
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export S2_TP="${S2_TP:-1}"
export S2_MODEL="${S2_MODEL:-Qwen/Qwen2.5-VL-72B-Instruct-AWQ}"
PORT="${S2_PORT:-8002}"
exec uvicorn server.system2_server:app --host 0.0.0.0 --port "$PORT"
