#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1
exec uvicorn server.system1_server:app --host 0.0.0.0 --port 8001
