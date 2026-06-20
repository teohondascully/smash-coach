# Smash Coach — Demo Day Runbook

Operating procedure for the live demo. Open this on your phone during setup
so you can refer without taking screen real estate from the demo.

## Hardware checklist

Before you sit down:

- [ ] Switch 2 + GuliKit Pro controller (charged)
- [ ] HDMI cable (Switch dock → Cam Link)
- [ ] Elgato Cam Link 4K
- [ ] USB-C → MacBook (NOT through a passive hub; Cam Link wants direct connection)
- [ ] USB-C → Gigabit Ethernet adapter (if venue has wired drops)
- [ ] MacBook charger
- [ ] 10ft HDMI cable so the rig isn't tangled

Plug-in order:
1. HDMI: Switch dock → Cam Link IN
2. USB-C: Cam Link → MacBook (direct port, not through hub if possible)
3. Confirm Cam Link status LED is solid

## Boot sequence

Order matters. Don't skip steps.

### 1. Cam Link enumeration check (30 sec)

```
./scripts/check_capture.sh
```

Should report a `Cam Link 4K` USB device and grab frames from device 0/1/2.
Open the saved snapshot to confirm which device index is the Switch feed.

### 2. Provision pod (~2 min)

```
prime wallet --plain                    # confirm balance > $20
prime availability list --gpu-type A100_80GB --gpu-count 2 --plain
./scripts/pod_up.sh                     # creates pod 72408b (A100x2 on-demand)
prime pods list --plain                 # confirm running, note pod-id
./scripts/cost_start.sh                 # marks cost-clock started
```

If `pod_up.sh` says the SKU is gone, edit `POD_ID` at the top of the script
to a fresh one from `prime availability list`.

### 3. Pod-side: download models + launch services (~15 min)

SSH in: `prime pods ssh <pod-id>`

On the pod:
```
nvidia-smi                              # confirm 2x A100 80GB
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd /workspace
git clone https://github.com/teohondascully/smash-coach.git
cd smash-coach
uv sync --extra server                  # ~3-5 min

# Models (~10 min, 56GB total)
export HF_TOKEN=<your-token>
mkdir -p /workspace/models
uv run huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir /workspace/models/qwen25-vl-7b --max-workers 8 &
uv run huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
  --local-dir /workspace/models/qwen25-vl-72b-awq --max-workers 8 &
wait

# Three tmux panes (or three SSH sessions):
# pane 1
./server/launch_ops_agent.sh
# pane 2
S1_MODEL=/workspace/models/qwen25-vl-7b S1_TP=1 CUDA_VISIBLE_DEVICES=0 \
  ./server/launch_system1.sh
# pane 3
S2_MODEL=/workspace/models/qwen25-vl-72b-awq S2_TP=1 CUDA_VISIBLE_DEVICES=1 \
  ./server/launch_system2.sh
```

S1 ready in ~30s. S2 takes 2–3 min (40GB AWQ load).

### 4. Mac-side: verify services + warm bench (~30 sec)

Get pod IP:
```
prime pods status <pod-id> --plain | grep -i 'ip\|addr'
export POD=<pod-ip>
```

Smoke:
```
curl http://$POD:9000/                # ops agent
curl http://$POD:8001/health          # S1
curl http://$POD:8002/health          # S2
```

If any return refused: try SSH tunneling instead:
```
prime pods ssh <pod-id> -- -L 8001:localhost:8001 -L 8002:localhost:8002 -L 9000:localhost:9000 -N &
export POD=localhost
```

Warm the prompt cache + verify latency:
```
S1_URL=http://$POD:8001/infer uv run python scripts/bench_s1.py
```
Expected: cold call ~1–3s, warm mean <100ms, cold/warm ratio >5x. If
warm mean >250ms, prefix caching isn't engaging — check the launch log
for `Automatic prefix caching is enabled`.

### 5. Open ops dashboard

```
./scripts/run_ops_dashboard.sh        # http://localhost:8502
```

Edit `data/ops_config.json` so `agent_url` = `http://$POD:9000` (or
`http://localhost:9000` if SSH-tunneling).

### 6. Launch demo

Two options.

**With backup recording (recommended for actual demo):**
```
POD=$POD RECORD_PATH=/tmp/demo_$(date +%s).mp4 ./scripts/run_demo.sh
```

**With debug dashboard (for engineering):**
```
POD=$POD DEBUG=1 ./scripts/run_demo.sh
```

Press `q` in the smash-coach window to quit. Press `d` to toggle debug
overlay during play.

## During the demo

- The HUD updates ~60Hz locally; action labels lag ~150ms (System 1
  poll). This is by design and invisible to the camera.
- Stock loss or damage spike triggers System 2 (~5–10s analysis). A
  rewind card window pops up with the counterfactual.
- If a rewind card looks weird, just press 'q' and relaunch. The pod
  servers stay up.

## Failure modes + recovery

| Symptom | Likely cause | Fix |
|---|---|---|
| HUD frozen at 0% damage, "neutral" actions | S1 unreachable | `curl http://$POD:8001/health` — if down, restart pane 2 |
| Rewind card never appears | S2 unreachable OR trigger never fires | `curl http://$POD:8002/health`; check stdout for "trigger:" lines |
| Cam Link goes black mid-demo | Switch slept (no HDMI passthrough) | Tap Joy-Con button to wake; preview keeps it alive |
| "OpenCV: not authorized to capture video" | macOS Privacy | System Settings → Privacy & Security → Camera → terminal |
| `prime pods list` shows pod stopped | Spot preemption or wallet drained | Provision new pod via `pod_up.sh`, redo step 3 |
| GPU OOM in S1 / S2 launch | Wrong TP or model path | Verify CUDA_VISIBLE_DEVICES and `nvidia-smi` |

## Shutdown

```
./scripts/cost_stop.sh                # stop cost clock
./scripts/pod_down.sh <pod-id>        # terminate pod (STOPS THE METER)
prime wallet --plain                  # confirm spend
```

**Always terminate the pod before sleeping.** It bills by the second.

## Stretch goals (only if time + budget permit)

- TTS narration on rewind card (set `OPENAI_API_KEY`; uncomment in `rewind_card.py`)
- 4× GPU scale-up (set `S1_TP=2 S2_TP=2`, two more A100s)
- Multi-character matchup (set `P1_CHAR` / `P2_CHAR` env)

## Files worth knowing about

- `data/ui_regions.json` — pixel rects for damage / stocks crop. Edit if
  Switch UI shifts on different output resolution.
- `data/action_vocab.json` — closed-vocab action labels for prompts.
- `server/prompts/system1.py` — S1 prompt + schema. Iterate here when
  the VLM mislabels actions.
- `data/frame_data.json` — scraped move data; cited by S2 in the rewind
  card. Re-runnable via `scripts/scrape_frame_data.py`.
