#!/usr/bin/env bash
# Mark the cost-tracker start time. Run right after the pod comes up.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -c "
from mac.ops import CostTracker
import json
cfg = json.load(open('data/ops_config.json'))
CostTracker('data/ops_state.json', cfg['rate_per_hour_usd'], cfg['budget_usd']).mark_started()
print(f'Cost clock started at \${cfg[\"rate_per_hour_usd\"]}/hr')
"
