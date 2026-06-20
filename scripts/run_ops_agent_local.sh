#!/usr/bin/env bash
# Run the ops_agent locally on the Mac for stub testing.
# On the real pod it runs via server/launch_ops_agent.sh.
# Stop with Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run uvicorn server.ops_agent:app --host 0.0.0.0 --port 9000
