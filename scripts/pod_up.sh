#!/usr/bin/env bash
# Provision the demo pod on Prime Intellect.
# Edit POD_ID at the top to switch between offerings.
# Usage: ./scripts/pod_up.sh
set -euo pipefail

POD_ID="${POD_ID:-72408b}"          # A100x2 on-demand, US, $3.30/hr
POD_NAME="${POD_NAME:-smashpod}"
POD_DISK="${POD_DISK:-1920}"
POD_IMAGE="${POD_IMAGE:-ubuntu_22_cuda_12}"

cd "$(dirname "$0")/.."

echo "=== Current wallet ==="
prime wallet --plain

echo
echo "=== Confirming SKU $POD_ID is still available ==="
prime availability list --gpu-type A100_80GB --gpu-count 2 --plain | grep "$POD_ID" || {
    echo "  $POD_ID no longer available."
    echo "  Re-run 'prime availability list --gpu-type A100_80GB --gpu-count 2 --plain' and update POD_ID."
    exit 1
}

echo
echo "=== Creating pod '$POD_NAME' from SKU $POD_ID ==="
echo "  this will spend money — Ctrl-C in 5s to abort"
sleep 5

prime pods create \
    --id "$POD_ID" \
    --name "$POD_NAME" \
    --disk-size "$POD_DISK" \
    --image "$POD_IMAGE" \
    --yes \
    --plain

echo
echo "=== Current pods ==="
prime pods list --plain

echo
echo "=== Starting cost clock ==="
uv run python -c "
from mac.ops import CostTracker
import json
cfg = json.load(open('data/ops_config.json'))
CostTracker('data/ops_state.json', cfg['rate_per_hour_usd'], cfg['budget_usd']).mark_started()
print(f'  Cost clock started at \${cfg[\"rate_per_hour_usd\"]}/hr')
"
