#!/usr/bin/env bash
set -euo pipefail
exec uvicorn server.ops_agent:app --host 0.0.0.0 --port 9000
