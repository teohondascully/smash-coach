#!/usr/bin/env bash
# Local end-to-end test: stub S1+S2 + live capture + recording.
# When this runs cleanly, the only thing left untested is the real VLM
# (which behaves identically to the stub as far as the Mac pipeline cares).
#
# Output: /tmp/dry_run_<timestamp>.mp4 with every HUD frame.
# Press 'q' in the smash-coach window to stop.
set -euo pipefail
cd "$(dirname "$0")/.."

RECORD_PATH="${RECORD_PATH:-/tmp/dry_run_$(date +%s).mp4}"
DEBUG="${DEBUG:-1}"

cleanup() {
    echo "[dry_run] stopping stubs..."
    kill $S1_PID $S2_PID 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[dry_run] starting stub S1 on :8001..."
uv run uvicorn server.stub_server:app --host 127.0.0.1 --port 8001 \
    >/tmp/dry_run_s1.log 2>&1 &
S1_PID=$!

echo "[dry_run] starting stub S2 on :8002..."
uv run uvicorn server.stub_server:app --host 127.0.0.1 --port 8002 \
    >/tmp/dry_run_s2.log 2>&1 &
S2_PID=$!

sleep 3
if ! curl -sf http://localhost:8001/health > /dev/null; then
    echo "[dry_run] stub S1 failed to start. Log:"
    tail -30 /tmp/dry_run_s1.log
    exit 1
fi
if ! curl -sf http://localhost:8002/health > /dev/null; then
    echo "[dry_run] stub S2 failed to start. Log:"
    tail -30 /tmp/dry_run_s2.log
    exit 1
fi

echo "[dry_run] stubs healthy. Launching main with RECORD_PATH=$RECORD_PATH"
echo "[dry_run] press 'q' in the smash-coach window to stop"
echo

RECORD_PATH="$RECORD_PATH" \
DEBUG="$DEBUG" \
S1_URL=http://localhost:8001/infer \
S2_URL=http://localhost:8002/counterfactual \
P1_CHAR=toon_link P2_CHAR=ike \
uv run python -m mac.main

echo
echo "[dry_run] done. recording at:"
echo "  $RECORD_PATH"
echo "  open $RECORD_PATH"
