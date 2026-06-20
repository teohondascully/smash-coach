#!/usr/bin/env bash
# Launch mac.main pointing at the GPU pod.
# Usage:
#   POD=<host[:port]>           ./scripts/run_demo.sh
#   POD=localhost DEBUG=1       ./scripts/run_demo.sh
#   CAP_DEV=0 POD=1.2.3.4       ./scripts/run_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

POD="${POD:-localhost}"
S1_PORT="${S1_PORT:-8001}"
S2_PORT="${S2_PORT:-8002}"

# Strip any trailing slash / scheme just in case.
POD_HOST="${POD#http://}"
POD_HOST="${POD_HOST#https://}"
POD_HOST="${POD_HOST%/}"

export S1_URL="${S1_URL:-http://${POD_HOST}:${S1_PORT}/infer}"
export S2_URL="${S2_URL:-http://${POD_HOST}:${S2_PORT}/counterfactual}"
export CAP_DEV="${CAP_DEV:-0}"
export S1_HZ="${S1_HZ:-7.0}"
export DEBUG="${DEBUG:-0}"
export P1_CHAR="${P1_CHAR:-toon_link}"
export P2_CHAR="${P2_CHAR:-ike}"

echo "=== smash coach demo ==="
echo "  POD       = $POD_HOST"
echo "  S1_URL    = $S1_URL"
echo "  S2_URL    = $S2_URL"
echo "  CAP_DEV   = $CAP_DEV"
echo "  S1_HZ     = $S1_HZ"
echo "  DEBUG     = $DEBUG"
echo "  P1 / P2   = $P1_CHAR / $P2_CHAR"
echo

exec uv run python -m mac.main
