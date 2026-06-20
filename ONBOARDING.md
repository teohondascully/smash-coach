# Smash Coach — Onboarding for Pair Programmer

> If you are Claude Code reading this for the first time: this file is the full project context. Read it top to bottom before doing anything. It supersedes anything else in the repo if there's a contradiction. The original spec lives at `docs/superpowers/specs/2026-06-19-smash-coach-design.md` and the implementation plan at `docs/superpowers/plans/2026-06-19-smash-coach.md` — those are the source of truth for architecture decisions.

## What we're building

A real-time AI coach + counterfactual replay engine for *Super Smash Bros. Ultimate* on the Nintendo Switch 2. The 24h hackathon goal: **Etched inference hackathon** in San Francisco, judges are founders of Anthropic, Cognition, Etched, Mercor.

Two visible artifacts during the demo:
- **Live AR HUD over the Switch feed**: damage % readouts, character bounding boxes, action labels, hitbox previews, threat zones. Renders at ~30-60fps depending on display path.
- **Post-event Counterfactual Rewind Card**: triggered on stock loss and high-damage exchanges. Shows the actual play side-by-side with one grounded alternative timeline, with frame-data citations ("Joker f-smash: 16f startup — spot-dodge would have opened a 9-frame punish window").

The matchup is locked: **Toon Link (P1, left) vs. Ike (P2, right) on Final Destination**, 1 stock. The teammate playing is on Toon Link.

## North star

The pitch: "we built a coach that watches Smash like a frame-perfect TO and explains losses in their own language." The technical flex is the architecture, not the silicon.

Three things the demo has to nail:
1. **Live HUD feels alive** — damage/action labels update visibly, threat circles track players.
2. **Rewind card is grounded** — quantitative claims (punish windows, frame advantage) come from a deterministic rule-based scorer over actual scraped frame data, not VLM hallucination.
3. **Inference-systems story** — dual-engine async architecture, grammar-constrained JSON decoding, prefix caching, multi-frame stack input. The judges from the inference world should see real optimization.

## Architecture

```
[Switch 2] --HDMI--> [Elgato Cam Link 4K] --USB--> [MacBook (player 1)]
                                                       |
                          ┌────────────────────────────┼────────────────────────┐
                          ▼                            ▼                        ▼
                  Local CV/OCR                    WebSocket / HTTP         HTTP trigger
                  (60Hz Tier 0/1)                 (5-10Hz JPEGs)           (on event)
                          │                            │                        │
                          │              [Prime Intellect pod, 2x A100 80GB]    │
                          │           ┌────────────────┴────────────┐           │
                          │           │  System 1 (1x A100)         │           │
                          │           │  Qwen2.5-VL-7B-Instruct     │           │
                          │           │  vLLM, TP=1                 │           │
                          │           │  → damage_pct, action_label,│           │
                          │           │    phase, intent (per JSON  │           │
                          │           │    schema, grammar enforced)│           │
                          │           └────────────────┬────────────┘           │
                          │                            │                        │
                          │           ┌────────────────┴────────────┐           │
                          │           │  System 2 (1x A100)         │◄──────────┘
                          │           │  Qwen2.5-VL-72B-AWQ         │
                          │           │  SGLang, TP=1               │
                          │           │  → counterfactual JSON +    │
                          │           │    frame-data citations     │
                          │           └────────────────┬────────────┘
                          │                            │
                          ▼                            ▼
                  [OpenCV HUD overlay]          [Rewind card UI]
```

### The pivots we made

When tracing through the codebase you'll see decisions that look weird. They're not — they're scars from real failures. Don't re-litigate without reading why:

- **OCR for damage was moved into the System 1 VLM output.** Tesseract and our home-grown template matcher both choked on Smash's outlined font under varying stage backgrounds. The VLM handles it for free at 5-10Hz, which is plenty for triggers and the HUD. Tier 0 OCR remains as a *backup* only for when S1 is unreachable.
- **8× H100 dropped to 2× A100_80GB** for hackathon budget reasons. $100 wallet; 2× A100 gives ~30h on-demand runway. The architecture supports TP scale-up via `S1_TP` / `S2_TP` env vars but we don't use it for the demo.
- **Matchup is Ike vs Toon Link, not Joker.** Joker data still exists in `data/{action_vocab,frame_data,hitboxes}.json` but `CHAR_MAP` defaults to `toon_link / ike`. Don't delete Joker; it's <1KB and demonstrates schema flexibility.
- **3-frame stack input to System 1** at ~100ms intervals. Single-frame phase detection (startup/active/endlag) was the weakest link; temporal context fixes it. See `mac/dispatcher.py`.
- **Prefix caching enabled on vLLM** (`enable_prefix_caching=True`). The system prompt is identical across every call, so cached prefix dominates per-call latency. Verify with `scripts/bench_s1.py` — cold/warm ratio should be >2x.

## What's built (state of the code right now)

All of these exist and have tests:

### Mac side (`mac/`)
- `capture.py` — UVC capture via cv2 with MJPG codec + buffer=1 for 60fps.
- `tier0_ocr.py` — local 60Hz OCR for **stocks only** (damage now comes from S1). Falls back to the template matcher at `mac/digit_match.py` and finally tesseract.
- `tier1_cv.py` — HSV-based character bbox detection.
- `dispatcher.py` — async HTTP client to S1 with rate limiting, multi-frame stack buffering, UI strip cropping, adaptive sampling on consecutive "neutral" responses.
- `state.py` — pydantic `StateT`, `StateBuffer`, `PlayerState`, `ActionState`.
- `frame_data.py` — load + lookup of scraped frame data, hitbox circles per move.
- `onset.py` — tracks when each player's action_label first changes (used for hitbox onset estimate).
- `trigger.py` — fires on stock loss or damage delta > 30% over 2s, with cooldown.
- `scorer.py` — deterministic frame-data scorer (punish windows, frame advantage).
- `hud.py` — OpenCV overlay renderer.
- `smoother.py` — 3-frame majority vote on action labels (kills flicker).
- `system2_client.py` — async client to S2 + saliency keyframe selection.
- `rewind_card.py` — post-event card UI (cv2-based, non-blocking via `RewindCardWindow` in main).
- `dashboard.py` — 4-panel cv2 debug dashboard (LIVE HUD / RAW CAPTURE / STATE / METRICS+LOGS).
- `ops.py` — `PodLifecycle` (prime CLI subprocess), `PodMetrics` (async httpx polling pod-side agent), `CostTracker`.
- `ops_dashboard.py` — Streamlit ops dashboard (`http://localhost:8502`).
- `main.py` — the orchestrator. Reads env vars: `CAP_DEV`, `S1_URL`, `S2_URL`, `S1_HZ`, `REWIND_SECS`, `TRIGGER_DELTA`, `TRIGGER_COOLDOWN`, `DEBUG`, `P1_CHAR`, `P2_CHAR`, `RECORD_PATH`.

### Server side (`server/`)
- `system1_server.py` — FastAPI + vLLM. POST `/infer` with `{images_b64: list, ts: list}` (multi-frame stack), returns JSON per the schema in `server/prompts/system1.py`. Heavy imports lazy.
- `system2_server.py` — FastAPI + SGLang. POST `/counterfactual`, returns counterfactual JSON.
- `ops_agent.py` — FastAPI on port 9000, exposes `/gpu` (nvidia-smi parsed), `/health/s{1,2}`, `/disk`.
- `stub_server.py` — local stub of both S1 and S2 endpoints for laptop dry runs. Returns schema-conformant fake responses with damage that drifts so triggers fire.
- `prompts/system1.py` — `build_system_prompt(p1_char, p2_char)`, `build_json_schema(p1_char, p2_char)`, action vocabulary loaded from `data/action_vocab.json`.
- `prompts/system2.py` — `COUNTERFACTUAL_SCHEMA`, `build_system_prompt(frame_data_blob)`.
- `launch_system1.sh`, `launch_system2.sh`, `launch_ops_agent.sh` — env-controlled launchers (`S1_TP`, `S1_MODEL`, `CUDA_VISIBLE_DEVICES`).

### Data (`data/`)
- `action_vocab.json` — canonical action labels per character (joker, toon_link, ike). The shared contract that prompts + scrape both consume.
- `frame_data.json` — scraped move data from ultimateframedata.com for all three characters.
- `hitboxes.json` — hand-rolled hitbox circles per move (fsmash, usmash so far).
- `ui_regions.json` — pixel rects for damage/stocks crops, hand-calibrated.
- `ops_config.json` — `rate_per_hour_usd`, `budget_usd`, `agent_url`, `poll_interval_s`.
- `digit_templates/` — 0-9, dot, pct PNG templates for the backup OCR.

### Scripts (`scripts/`)
- `preflight.sh` — run before every demo or dry run. Walks every dep + checks status.
- `dry_run.sh` — local end-to-end test: stub S1 + S2 + mac.main + MP4 recording.
- `run_demo.sh` — main launcher. Wraps env vars: `POD=<host> ./scripts/run_demo.sh`.
- `run_stub.sh` — start stub S1+S2 on :8001/:8002.
- `run_ops_dashboard.sh` — Streamlit ops dashboard.
- `run_ops_agent_local.sh` — start ops agent on Mac for stub testing.
- `pod_up.sh`, `pod_down.sh`, `cost_start.sh`, `cost_stop.sh` — pod lifecycle.
- `bench_s1.py` — latency benchmark, cold/warm ratio to verify prefix caching.
- `preview.sh` — live Cam Link preview (keeps Switch awake since Cam Link has no HDMI passthrough).
- `capture_corpus.sh` — auto-capture frames at fixed interval, used to build the sample fixture corpus.
- `check_capture.sh` — diagnose Cam Link USB enumeration.
- `scrape_frame_data.py` — re-scrapes ultimateframedata.com for all characters.

### Tests (`tests/`)
34 passing tests covering state, frame data, scorer, trigger, onset, prompts schema. Live capture frames at `tests/fixtures/sample_*.jpg` (76 frames from real Switch capture — these drive prompt iteration).

## How to run things (cheat sheet)

```
./scripts/preflight.sh                           # always run first
./scripts/dry_run.sh                             # local end-to-end test (stub VLM)
POD=<ip> ./scripts/run_demo.sh                   # real demo against pod
POD=<ip> DEBUG=1 ./scripts/run_demo.sh           # demo + debug dashboard
RECORD_PATH=/tmp/x.mp4 POD=<ip> ./scripts/run_demo.sh  # demo + MP4 recording
./scripts/run_ops_dashboard.sh                   # Streamlit ops at :8502
S1_URL=http://<pod>:8001/infer uv run python scripts/bench_s1.py  # S1 latency bench
```

## Pod context (CRITICAL — read before doing anything that costs money)

- The pod ID is in `$POD_ID` on whoever's machine (currently in player 1's shell).
- **The wallet is $100 on Personal account.** Player 2 SSHing in does NOT incur separate billing. Pod cost is purely wall-clock × $3.30/hr.
- **Always know how to terminate the pod.** `./scripts/pod_down.sh <pod-id>` or `prime pods terminate <pod-id> --yes --plain`. Idle pod still bills.
- The actual current pod runs on Crusoe Cloud, US, 2× A100 80GB PCIe.
- Data lives at `/ephemeral/` (1.7TB), models at `/ephemeral/models/qwen25-vl-{7b,72b-awq}`.
- Servers run as plain `uvicorn` in tmux/screen panes on the pod (no systemd).
- Logs are stdout. If a server crashes, check the pane.

### How you (Player 2) join Player 1's existing pod

We do NOT have you provision your own pod. You share Player 1's pod via shared API key + SSH. Two-step setup:

**1. Install + authenticate Prime Intellect CLI** (gives you pod listing, status, ssh, terminate):

```bash
uv tool install -U prime
prime login
# Paste Player 1's API key when prompted (he'll DM it to you).
```

Verify:
```bash
prime whoami --plain         # should show Player 1's account
prime pods list --plain      # should show 'smashpod' RUNNING
prime wallet --plain         # see remaining $; check this often
```

**2. SSH in.** Two equivalent ways:

```bash
# via prime (auto-uses Player 1's pod registration):
prime pods ssh <pod-id>

# or directly with the IP, after Player 1 appends your public key to
# ~/.ssh/authorized_keys on the pod:
ssh ubuntu@<pod-ip>
```

The first form (`prime pods ssh`) will Just Work because you're authenticated as Player 1 — no separate key wrangling needed. The second is the backup if the prime CLI gets weird.

### How to safely terminate the pod

Yes, you can stop the meter — please do if you're the last one to walk away from it.

```bash
# what's currently spinning?
prime pods list --plain
prime pods status <pod-id> --plain
prime wallet --plain               # current spend

# stop the meter:
prime pods terminate <pod-id> --yes --plain
```

Or use the bundled script:

```bash
./scripts/pod_down.sh <pod-id>
```

**Coordinate before terminating.** Drop a message in chat saying "terminating pod in 60s, stop me if you're using it" — gives Player 1 a chance to halt you if he's mid-demo-prep. The pod's $3.30/hr; mid-task interruption is annoying but not catastrophic. Re-provisioning takes ~5 min via `./scripts/pod_up.sh` and re-downloading the 32GB AWQ model takes another ~10 min, so don't terminate casually if either of you is actively using S1/S2.

Rule of thumb: **if no one will touch it in the next 30 minutes, terminate.**

## Player 2 setup (no Switch, no Cam Link)

You don't have the Switch. To exercise the pipeline locally, **use a pre-recorded MP4 of gameplay** as the capture source. Player 1 will send you `gameplay.mp4` separately.

### Your one job: add a `VideoFileCapture` mode

Currently `mac/capture.py:Capture` takes `device_index: int` and opens a UVC device. Add an alternate path:

- If `CAP_DEV` is a path like `/path/to/gameplay.mp4`, open it as a video file via `cv2.VideoCapture(path)`.
- Yield frames at the file's natural FPS (use the video's `CAP_PROP_FPS`).
- Loop the video when it ends, OR exit cleanly — your call.

A clean diff would be:

```python
# in mac/capture.py
def __init__(self, device_index_or_path, ...):
    if isinstance(device_index_or_path, str) and os.path.exists(device_index_or_path):
        self.cap = cv2.VideoCapture(device_index_or_path)
    else:
        self.cap = cv2.VideoCapture(int(device_index_or_path), cv2.CAP_AVFOUNDATION)
        # ...existing UVC config
```

And in `mac/main.py`, parse `CAP_DEV` as `int(cap_dev) if cap_dev.isdigit() else cap_dev`.

With that change, you can run:

```bash
CAP_DEV=/path/to/gameplay.mp4 POD=<pod-ip> ./scripts/run_demo.sh
```

…and the entire pipeline runs against the recorded video instead of live capture. Same VLM, same triggers, same rewind card.

### What you can iterate on without burning pod $

The pod-burn-rate concern means you should NOT keep the pod busy with idle exploration. Things to do that are zero-cost:

1. **Prompt iteration on System 1.** `tests/fixtures/sample_178*.jpg` is a real Switch capture corpus. Read sample frames into a small script, post them to the stub server (or the real one for ground truth), check what the model returns vs what we know is true. Tune `server/prompts/system1.py` to make labels more accurate. The stub has fake responses — you'll need real S1 for real ground truth (coordinate with player 1 for a short time window with S1 up).
2. **Rewind card visual polish.** `mac/rewind_card.py:render_card`. Make it look like something a judge would screenshot. Currently functional but visually basic.
3. **The latency overlay on judge view.** Small widget in the corner of `mac.main` showing "S1 12ms" / "S2 idle" — inference-hackathon judges literally love this.
4. **Few-shot examples in S1 prompt.** Use the corpus to add 2-3 frame→label examples directly in the prompt body. Should sharpen labels.

## Cost watchdog (read this twice)

Pod cost is real. ~$3.30/hr. **DO NOT**:
- Leave pod up overnight
- Run `bench_s1.py` with `N=10000`
- Spin up extra services without thinking about VRAM
- Make a decision that requires re-downloading the 32GB AWQ model

**DO**:
- Use the local stub (`./scripts/run_stub.sh`) for pipeline testing
- Use the corpus frames for prompt evaluation
- Check `prime wallet --plain` periodically
- Run `./scripts/cost_stop.sh && ./scripts/pod_down.sh <id>` before walking away

## Open questions / things still in flight (as of last commit)

- Pod is provisioning / model downloads underway. S1 + S2 servers not yet launched.
- Prompt iteration on S1 hasn't happened against real S1 yet.
- The live HUD has been tested end-to-end against the **stub** S1+S2, not the real one yet.
- The demo pitch hasn't been written.

## Where to look in the codebase if you're confused

- "How does data flow from capture to HUD?" → `mac/main.py` is the single orchestrator. Read top to bottom.
- "What does the VLM output look like?" → `server/prompts/system1.py::build_json_schema`.
- "What's in s_t?" → `mac/state.py:StateT`.
- "When does System 2 fire?" → `mac/trigger.py:TriggerDetector`.
- "How are frame-data citations grounded?" → `mac/scorer.py:score_counterfactual` + the citations field in the S2 schema.
- "Why is this so complex?" → it's not, actually. The complexity is mostly graceful-degradation glue. Each component does one thing.

## Git / repo conventions

- `main` branch only. Frequent small commits.
- Conventional commit prefixes: `feat(area)`, `fix`, `chore`, `docs`.
- Commit includes `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` (this is just the team convention).
- Push after every commit.
- Don't run destructive git ops without explicit asking.

## Files NOT to touch

- `data/action_vocab.json` — the shared contract between Thread A (prompts) and Thread B (frame data). If you must change, coordinate.
- `data/frame_data.json` — re-run `scripts/scrape_frame_data.py` to regenerate.
- `data/hitboxes.json` — hand-rolled, fine to extend.
- `pyproject.toml` requires-python pin — if you change it, both Macs and the pod re-sync.

## Quick start for your Claude Code

1. Read this whole file.
2. Run `./scripts/preflight.sh` — see what's installed and what isn't on your machine.
3. Run `./scripts/dry_run.sh` (if you have the gameplay.mp4 and have added the VideoFileCapture mode) — see the pipeline run.
4. Ask Player 1 in the team chat for: pod-ip, current pod-id, current cost, any open blockers.
5. Pick one of the "what you can iterate on" tasks above and PR it.

Good luck. Don't burn the budget.
