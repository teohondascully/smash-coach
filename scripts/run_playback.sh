#!/usr/bin/env bash
# Play back the recorded clip with PRECOMPUTED labels — no pod, no tunnel.
# 72B-accurate labels, perfectly synced, full speed. Generate the track first:
#   scripts/precompute_labels.py  (then pull labels_track.json to data/)
#
# Usage:
#   ./scripts/run_playback.sh
#   DEBUG=1 ./scripts/run_playback.sh        # + 4-panel dashboard
#   RECORD_PATH=/tmp/demo.mp4 ./scripts/run_playback.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export CAP_DEV="${CAP_DEV:-data/gameplay.mov}"
export CAP_SKIP_SECS="${CAP_SKIP_SECS:-11}"
export LABEL_TRACK="${LABEL_TRACK:-data/labels_track.json}"
export P1_CHAR="${P1_CHAR:-toon_link}"
export P2_CHAR="${P2_CHAR:-ike}"
export DEBUG="${DEBUG:-0}"
export RECORD_PATH="${RECORD_PATH:-}"

echo "=== smash coach — recorded playback (precomputed labels, no pod) ==="
echo "  CAP_DEV     = $CAP_DEV  (skip ${CAP_SKIP_SECS}s intro)"
echo "  LABEL_TRACK = $LABEL_TRACK"
echo "  P1 / P2     = $P1_CHAR / $P2_CHAR    DEBUG=$DEBUG"
echo "  (press 'q' in the window to quit)"
echo

if [ ! -f "$LABEL_TRACK" ]; then
  echo "ERROR: $LABEL_TRACK not found."
  echo "  Generate it with scripts/precompute_labels.py against a running 72B,"
  echo "  then copy it to $LABEL_TRACK."
  exit 1
fi

exec uv run python -m mac.main
