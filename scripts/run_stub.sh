#!/usr/bin/env bash
# Run the S1+S2 stub on ports 8001 and 8002 so mac.main can be exercised
# end-to-end on a Mac without the GPU pod.
#
# Two processes (one per port) so the trigger dispatch path is real:
# main posts to :8001 for S1, :8002 for S2.
set -euo pipefail
cd "$(dirname "$0")/.."

cleanup() {
    kill $S1_PID $S2_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== stub S1 on :8001 ==="
uv run uvicorn server.stub_server:app --host 0.0.0.0 --port 8001 &
S1_PID=$!

echo "=== stub S2 on :8002 ==="
uv run uvicorn server.stub_server:app --host 0.0.0.0 --port 8002 &
S2_PID=$!

sleep 2
echo
echo "stubs running. To exercise mac.main against them:"
echo "  POD=localhost ./scripts/run_demo.sh"
echo
echo "Ctrl-C to stop both."
wait
