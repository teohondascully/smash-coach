#!/usr/bin/env bash
# Run the live HUD against the REAL System 1 on the pod, off a recorded clip.
#
# The pod's inference ports are firewalled from the public internet, so this
# script opens an SSH tunnel (localhost:8001 -> pod:8001, +8002/9000) if one
# isn't already up, then launches mac.main pointed at the tunnel.
#
# S2 (rewind card) may not be running yet — that's fine, the HUD degrades
# gracefully (triggers fire, no card shown).
#
# Usage:
#   ./scripts/run_live_s1.sh                       # uses data/gameplay.mov
#   CAP_DEV=0 ./scripts/run_live_s1.sh             # live capture card instead
#   POD_IP=1.2.3.4 ./scripts/run_live_s1.sh        # different pod IP
#   DEBUG=1 ./scripts/run_live_s1.sh               # + 4-panel debug dashboard
set -euo pipefail
cd "$(dirname "$0")/.."

POD_IP="${POD_IP:-204.52.24.178}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
SSH_USER="${SSH_USER:-ubuntu}"
S1_PORT="${S1_PORT:-8001}"
S2_PORT="${S2_PORT:-8002}"
OPS_PORT="${OPS_PORT:-9000}"
CAP_DEV="${CAP_DEV:-data/gameplay.mov}"
# gameplay.mov opens on ~10s of noise + character-select; skip to the match.
CAP_SKIP_SECS="${CAP_SKIP_SECS:-11}"

# Gate on the actual S1 health endpoint through the tunnel — a listening socket
# alone can be a stale/half-dead tunnel, so we test end-to-end reachability.
s1_healthy() {
  curl -s -m 6 "http://localhost:${S1_PORT}/health" 2>/dev/null | grep -q '"ok":true'
}

# --- ensure a WORKING tunnel is up -----------------------------------------
if s1_healthy; then
  echo "[tunnel] S1 already reachable on localhost:${S1_PORT} — reusing tunnel"
else
  echo "[tunnel] (re)opening SSH tunnel to ${SSH_USER}@${POD_IP} ..."
  # Drop any stale tunnel holding our local ports, then open a fresh one with
  # keepalives so it survives idle periods.
  pkill -f "${S1_PORT}:localhost:${S1_PORT}" 2>/dev/null || true
  sleep 1
  ssh -i "$SSH_KEY" -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -fN \
    -L "${S1_PORT}:localhost:${S1_PORT}" \
    -L "${S2_PORT}:localhost:${S2_PORT}" \
    -L "${OPS_PORT}:localhost:${OPS_PORT}" \
    "${SSH_USER}@${POD_IP}"
  sleep 2
fi

# --- verify S1 is actually answering ---------------------------------------
echo -n "[check] S1 /health ... "
if s1_healthy; then
  echo "OK (real System 1 reachable)"
else
  echo "NOT READY"
  echo "  S1 isn't answering on localhost:${S1_PORT}. Is the s1 tmux session up on the pod?"
  echo "  On the pod: tmux ls ; tail -f /tmp/s1.log"
  exit 1
fi

# --- launch the live HUD ----------------------------------------------------
export CAP_DEV
export CAP_SKIP_SECS
# Look-ahead sync: delay display by ~the S1 round-trip so labels line up with
# the action. Set to 0 for live (un-synced) playback.
export DISPLAY_DELAY="${DISPLAY_DELAY:-2.0}"
export S1_URL="http://localhost:${S1_PORT}/infer"
export S2_URL="http://localhost:${S2_PORT}/counterfactual"
export S1_HZ="${S1_HZ:-7.0}"
export DEBUG="${DEBUG:-0}"
export P1_CHAR="${P1_CHAR:-toon_link}"
export P2_CHAR="${P2_CHAR:-ike}"
export RECORD_PATH="${RECORD_PATH:-}"

echo "=== smash coach — live HUD on real S1 ==="
echo "  CAP_DEV = $CAP_DEV   (skip ${CAP_SKIP_SECS}s intro)"
echo "  S1_URL  = $S1_URL"
echo "  S1_HZ   = $S1_HZ   DEBUG = $DEBUG   P1/P2 = $P1_CHAR/$P2_CHAR"
echo "  (press 'q' in the smash-coach window to quit)"
echo

exec uv run python -m mac.main
