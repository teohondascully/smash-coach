#!/usr/bin/env bash
# Pre-demo sanity check. Walks every dependency and prints PASS/FAIL.
# Run before each dry-run + immediately before going live.
set -uo pipefail
cd "$(dirname "$0")/.."

pass() { printf "  \033[32m✓\033[0m  %s\n" "$1"; }
warn() { printf "  \033[33m!\033[0m  %s\n" "$1"; ((WARN++)) || true; }
fail() { printf "  \033[31m✗\033[0m  %s\n" "$1"; ((FAIL++)) || true; }

FAIL=0
WARN=0

echo "=== smash coach pre-flight ==="

# --- env / deps ---
echo
echo "== environment =="
if [ -d .venv ]; then pass "uv venv exists"; else fail ".venv missing — run 'uv sync'"; fi

if uv run python -c "import cv2, numpy, fastapi, httpx, pydantic" 2>/dev/null; then
    pass "core python deps importable"
else
    fail "core python deps missing — run 'uv sync'"
fi

if uv run python -c "import streamlit" 2>/dev/null; then
    pass "streamlit installed (ops dashboard)"
else
    warn "streamlit missing — ops dashboard won't work"
fi

# --- repo files ---
echo
echo "== repo state =="
for f in data/ui_regions.json data/frame_data.json data/hitboxes.json \
         data/action_vocab.json data/ops_config.json; do
    if uv run python -c "import json; json.load(open('$f'))" 2>/dev/null; then
        pass "$f valid JSON"
    else
        fail "$f missing or invalid"
    fi
done

if [ -d data/digit_templates ] && ls data/digit_templates/*.png &>/dev/null; then
    pass "digit templates present"
else
    warn "digit templates missing (backup OCR off)"
fi

# --- module imports ---
echo
echo "== module imports =="
for mod in mac.capture mac.dispatcher mac.main mac.ops mac.dashboard \
          server.system1_server server.system2_server server.stub_server \
          server.ops_agent; do
    if uv run python -c "import $mod" 2>/dev/null; then
        pass "import $mod"
    else
        fail "import $mod failed"
    fi
done

# --- tests ---
echo
echo "== test suite =="
if out=$(uv run pytest -q tests/ 2>&1); then
    pass "$(echo "$out" | tail -1)"
else
    fail "pytest failed: $(echo "$out" | tail -3 | head -1)"
fi

# --- hardware ---
echo
echo "== hardware =="
if system_profiler SPUSBDataType 2>/dev/null | grep -qi 'cam link\|elgato'; then
    pass "Cam Link enumerated on USB"
else
    warn "Cam Link not seen on USB (plug in before running mac.main)"
fi

# --- external tools ---
echo
echo "== external tools =="
if command -v prime &>/dev/null; then
    pass "prime CLI on PATH"
    if prime whoami --plain 2>/dev/null | grep -q '@\|user\|name'; then
        pass "prime CLI authenticated"
    else
        warn "prime CLI not logged in — run 'prime login'"
    fi
    if balance=$(prime wallet --plain 2>/dev/null | grep -i balance); then
        pass "prime wallet readable: $balance"
    fi
else
    warn "prime CLI not installed (pod provisioning will fail)"
fi

if command -v gh &>/dev/null; then
    if gh auth status &>/dev/null; then
        pass "gh CLI authenticated"
    else
        warn "gh CLI not authenticated"
    fi
else
    warn "gh CLI missing (commits work but pushes need it)"
fi

if [ -n "${HF_TOKEN:-}" ]; then
    pass "HF_TOKEN set in env"
else
    warn "HF_TOKEN not set (needed on pod for model downloads)"
fi

# --- summary ---
echo
echo "=== summary ==="
if [ "$FAIL" -gt 0 ]; then
    echo "  $FAIL failures, $WARN warnings  — fix failures before going live"
    exit 1
elif [ "$WARN" -gt 0 ]; then
    echo "  0 failures, $WARN warnings  — likely fine, review warnings"
    exit 0
else
    echo "  all green"
    exit 0
fi
