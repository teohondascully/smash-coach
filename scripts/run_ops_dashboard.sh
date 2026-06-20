#!/usr/bin/env bash
# Open the Streamlit ops dashboard in your browser.
# Stop with Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
exec uv run streamlit run mac/ops_dashboard.py --server.port 8502
