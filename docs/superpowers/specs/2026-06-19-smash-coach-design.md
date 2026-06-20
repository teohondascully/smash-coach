# Smash Coach: Live AR Coach + Counterfactual Replay Engine

**Date:** 2026-06-19
**Context:** Etched inference hackathon, 24h, San Francisco. 8× H100 NVLink cluster on Prime Intellect. Judges: founders of Anthropic, Cognition, Etched, Mercor. Track: Applied AI / Talent Marketplace.

## North Star

A live AR coaching system + post-event counterfactual replay engine for *Super Smash Bros. Ultimate*, demoed by two players (the team) in front of the panel. The system has two visible surfaces:

1. **Live AR HUD** rendered over the captured game feed: threat zones, hitbox previews, ledge danger lines, damage % readouts, action labels and intent estimates.
2. **Post-event Counterfactual Rewind Card**: triggered on stock loss and significant in-match exchanges. Shows the actual play side-by-side with one grounded alternative timeline, citing real frame data.

The combined demo positions the system as both an **AI coach as a service** (Anthropic/Cognition resonance) and an **inference-systems flex** showcasing what 6 H100s + a 70B-class VLM enable (Etched resonance).

## Demo Arc (90 seconds)

- **0:00–0:15** — Players boot the match. Judges see the MacBook output: live game + AR HUD. Deterministic overlays (%, stocks, ledge proximity) update at 60Hz; VLM-derived overlays (threat zones, action labels, intent) update at ~5–10Hz.
- **0:15–0:60** — Normal play. Text overlay captions surface coaching notes every 5–10s ("you're rolling behind every shield pressure string").
- **0:60–0:75** — Stock loss. System 2 has been analyzing the pre-death buffer. A **Counterfactual Rewind Card** appears: actual death replayed alongside one alternative timeline, with frame-data citations ("Joker f-smash: 16f startup, 38f total — spot-dodge on frame 140 would have given you a 9-frame punish window").
- **0:75–0:90** — Match resumes / debrief.

## Locked Scope

| Area | Decision |
|---|---|
| Track framing | "AI Smash Coach" + "Counterfactual Replay" blend |
| Match | Joker vs. Toon Link, Final Destination, 1 stock |
| Players | Team members on GuliKit Pro controllers (Bluetooth → Switch 2) |
| Coach modality | Text overlay primary; TTS post-stock as stretch goal |
| Controller telemetry | Out of scope. Schema accepts future `controller_input_t` field but no live tap (Bluetooth-only controllers preclude clean Mac-side capture without extra hardware) |
| Hitboxes | In scope, approximated from frame data + character anchor + action label + time-since-onset |
| Frame data grounding | In-prompt JSON blob (~60 entries for two characters), no vector DB |
| Counterfactual numbers | Qualitative bands from VLM; quantitative claims (frame advantage, punish windows) computed by a rule-based scorer over frame data |

## Architecture

```
[Switch 2] --HDMI--> [Elgato HD60 X] --USB--> [MacBook]
                                                |
                          ┌─────────────────────┼─────────────────────┐
                          ▼                     ▼                     ▼
                  Local CV/OCR              WebSocket            HTTP trigger
                  (60Hz Tier 0/1)           (5-10Hz JPEGs)       (on event)
                          │                     │                     │
                          │              [Prime Intellect 8x H100]    │
                          │           ┌─────────┴──────────┐          │
                          │           │   System 1 (2x)    │          │
                          │           │  Qwen2.5-VL-7B     │          │
                          │           │  vLLM, TP=2        │          │
                          │           │  → action labels,  │          │
                          │           │    intent, JSON s_t│          │
                          │           └────────┬───────────┘          │
                          │                    │                      │
                          │           ┌────────┴───────────┐          │
                          │           │   System 2 (4x)    │◄─────────┘
                          │           │  Qwen2.5-VL-72B    │
                          │           │  SGLang, TP=4      │
                          │           │  + spec-decode     │
                          │           │   draft (2x spare) │
                          │           │  → counterfactual  │
                          │           │    JSON + citations│
                          │           └────────┬───────────┘
                          │                    │
                          ▼                    ▼
                  [OpenCV HUD overlay]   [Rewind card UI]
```

## Components

### Mac client
- **Capture:** Elgato HD60 X as UVC device, OpenCV `VideoCapture`. Target 60Hz raw frame loop.
- **Tier 0 deterministic CV:** template-match / tiny CNN for damage % per player, stock icons, timer. Hardcoded UI regions for FD. Runs every frame, sub-millisecond.
- **Tier 1 lightweight CV:** character bounding boxes (small finetuned YOLO or color-keyed segmentation on costumes), facing direction, ledge proximity, airborne/grounded (y vs. known FD geometry), velocity from frame delta.
- **Frame dispatcher:** downsamples to 640×640 JPEG at quality ~70, throttles to 5–10Hz, ships over WebSocket to System 1.
- **State buffer:** rolling 10s window of $s_t$ dicts, indexed by timestamp. Source of truth for System 2 triggers.
- **Event detector:** local rules over $s_t$ stream — stock-loss, damage-delta > 30% in <2s, punish window opened/closed. On event, packages last 5s of $s_t$ + 5–10 saliency keyframes and HTTP-POSTs to System 2.
- **HUD renderer:** OpenCV draws overlays in a separate compositing layer. Stale-tolerant — last-known $s_t$ stays on screen if System 1 latency spikes.
- **Rewind card UI:** lightweight HTML or Pygame surface. Receives System 2 JSON, renders side-by-side replay + citation chips.

### System 1 — Micro-Perception Engine
- **Hardware:** 2× H100, TP=2.
- **Model:** Qwen2.5-VL-7B.
- **Server:** vLLM with `--enforce-eager` off, grammar-constrained JSON output.
- **Input:** single 640×640 JPEG per call.
- **Output:** JSON $s_t$ Tier 2 fields per player: `action_label` (closed vocabulary, ~20 actions per character), `action_phase` (startup / active / endlag / neutral), `intent` (closed vocabulary: pressuring / ledge-trapping / neutral / recovering / punishing), per-field confidence.
- **Throughput target:** 5–10 Hz sustained.

### System 2 — Macro-Strategic Counterfactual Engine
- **Hardware:** 4× H100, TP=4, plus 2× H100 for speculative-decoding draft model (stretch).
- **Model:** Qwen2.5-VL-72B AWQ.
- **Server:** SGLang with JSON-schema grammar enforcement.
- **Input per call:**
  1. System prompt with role + instructions.
  2. Frame data blob (full ~60-entry JSON for Joker + Toon Link inline).
  3. $s_t$ trajectory: last 5s, ~25–50 timestamped state dicts.
  4. 5–10 saliency keyframes (selected by the Mac at each `action_onset` event in the window).
  5. Output schema: `{summary, chosen_action, counterfactual_action, frame_data_citations[], qualitative_likelihood}`.
- **Quantitative scoring:** a separate Python rule-based scorer ingests `counterfactual_action` + frame data, computes punish window, frame advantage, range deltas. These deterministic numbers are what the rewind card displays.
- **Latency target:** <8s per call; runs async, never blocks live HUD.

### Frame Data Module
- Static `frame_data.json` scraped from ultimateframedata.com for Joker and Toon Link.
- Per move: `name, startup_f, active_f, endlag_f, landing_lag_f, shield_advantage, on_hit_advantage, range_estimate, category`.
- Loaded once on both Mac (for scorer) and System 2 (inlined in prompt).

### Hitbox Approximation
- Per move, store `hitbox_offsets[]` (a list of circles with `(dx, dy, radius, active_frame_range)`) relative to character anchor.
- At render time, the Mac joins `character_position + facing × hitbox_offsets` for the currently-active move, looking up the active-frame window from frame data and using `time_since_action_onset` (estimated from System 1's `action_phase` transitions) as the index.
- Because onset estimation is ±100–200ms without controller telemetry, hitbox accuracy is best on signposted moves (smashes, specials, projectiles) and degrades on fast aerials. The HUD shows a confidence indicator on hitbox overlays.

## State Representation $s_t$

```
s_t = {
  t: float,                     # seconds since match start
  damage: {p1: float, p2: float},
  stocks: {p1: int, p2: int},
  positions: {
    p1: {x, y, facing, airborne, vx, vy},
    p2: {x, y, facing, airborne, vx, vy}
  },
  actions: {
    p1: {label, phase, confidence, onset_estimate_t},
    p2: {label, phase, confidence, onset_estimate_t}
  },
  intent: {p1, p2},             # closed vocab
  derived: {
    distance, relative_facing,
    ledge_owner, stage_control_estimate,
    active_punish_window_for: p1|p2|null
  },
  controller_input_t: null      # reserved for future
}
```

## Failure Modes & Mitigations

| Failure | Mitigation |
|---|---|
| System 2 slower than expected | Rewind card shows skeleton "thinking..." state, populates async. Never blocks HUD. |
| System 1 latency spike | Last-known $s_t$ persists on HUD; no flicker. |
| VLM hallucinates invalid action | Grammar-constrained JSON limits to closed vocabulary; invalid → "unknown", fall through. |
| Network drops to H100 node | Tier 0/1 local CV continues. HUD shows %, positions, ledge lines even with cloud down. |
| Hitbox onset desync | Confidence indicator; degrade to "range circles only" if action-phase signal is unreliable. |
| Counterfactual probability hallucination | VLM gives qualitative band only; numbers come from deterministic rule-based scorer over frame data. |

## 24h Schedule

- **H0–H2 — Infra bring-up (parallel).** Mac client skeleton + capture + WS scaffolding (player A). vLLM Qwen2.5-VL-7B + SGLang Qwen2.5-VL-72B-AWQ stood up on cluster (player B). Smoke test with sample frame.
- **H2–H6 — Tier 0/1 perception + frame data scrape.** Damage OCR, stock counter, character bbox. Frame data JSON for Joker + Toon Link. $s_t$ pydantic schema.
- **H6–H12 — System 1 + live HUD.** Closed-vocab prompting until action labels are reliable on top ~20 actions per character. HUD renders threat zones, ledge lines, %, action captions. **Cut line @ H12:** if action labels unreliable, drop per-move hitbox polygons to range circles only.
- **H12–H18 — System 2 + rewind card.** Mac trigger logic. System 2 prompt + scorer. Rewind card UI with side-by-side + citation chips. **Cut line @ H18:** if card UI runs late, static card with thumbnails + text, no replay video.
- **H18–H21 — Polish, integration, dry-run.** End-to-end match runs. Latency audit; fix anything >100ms in live loop.
- **H21–H23 — Stretch.** TTS post-stock voice; speculative decoding on System 2 (only if SGLang spec-decode wiring looks <2h).
- **H23–H24 — Rehearsal + buffer.** 3+ full run-throughs.

## Out of Scope (explicit YAGNI)

- World model / RL of any kind
- Multi-character generalization (schema supports it; not tested or tuned)
- Stages other than FD
- Audio analysis from game sound
- Player skill rubric ("scout-a-pro" output)
- Vector DB / actual RAG
- Controller telemetry day-of
- Hotswap between models

## Hardware Shopping List (pre-hackathon)

- Elgato HD60 X (low-latency UVC capture)
- USB-C → Gigabit Ethernet adapter (drops jitter if venue has wired drops)
- Powered USB-C hub with HDMI passthrough
- 10ft HDMI cable
- (Optional) small portable monitor for the rewind card surface
