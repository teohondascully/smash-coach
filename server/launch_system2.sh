#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2,3,4,5
exec uvicorn server.system2_server:app --host 0.0.0.0 --port 8002
