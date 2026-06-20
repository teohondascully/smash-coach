#!/usr/bin/env bash
# Mark the cost-tracker stop time. Run after pod terminate succeeds.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -c "
from mac.ops import CostTracker
import json
cfg = json.load(open('data/ops_config.json'))
ct = CostTracker('data/ops_state.json', cfg['rate_per_hour_usd'], cfg['budget_usd'])
spent = ct.spent_usd()
ct.mark_stopped()
print(f'Cost clock stopped. Spent this session: \${spent:.2f}')
"
