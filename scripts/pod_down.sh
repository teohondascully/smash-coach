#!/usr/bin/env bash
# Terminate the demo pod. Stops the meter.
# Usage: ./scripts/pod_down.sh [pod-id]
# If no pod-id passed, terminates the first running pod named 'smashpod'.
set -euo pipefail

cd "$(dirname "$0")/.."

POD_ID="${1:-}"

if [[ -z "$POD_ID" ]]; then
    echo "=== Looking up running pods ==="
    prime pods list --plain
    echo
    echo "No pod-id passed. Usage: ./scripts/pod_down.sh <pod-id>"
    exit 1
fi

echo "=== Stopping cost clock ==="
uv run python -c "
from mac.ops import CostTracker
import json
cfg = json.load(open('data/ops_config.json'))
ct = CostTracker('data/ops_state.json', cfg['rate_per_hour_usd'], cfg['budget_usd'])
spent = ct.spent_usd()
ct.mark_stopped()
print(f'  Spent this session (local clock): \${spent:.2f}')
" || echo "  (cost tracker unavailable; continuing)"

echo
echo "=== Terminating pod $POD_ID ==="
prime pods terminate "$POD_ID" --yes --plain

echo
echo "=== Final wallet ==="
prime wallet --plain
