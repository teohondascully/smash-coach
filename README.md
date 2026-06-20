# Smash Coach

Real-time AI coach + counterfactual replay engine for *Super Smash Bros. Ultimate*, running on an 8× H100 NVLink cluster. Built for the Etched inference hackathon.

## Architecture

Dual-engine async system:

- **System 1 (perception, 2× H100):** Qwen2.5-VL-7B via vLLM, TP=2. Polled at 5–10 Hz with structured-JSON output for action labels, phases, intent.
- **System 2 (counterfactual, 4× H100):** Qwen2.5-VL-72B-AWQ via SGLang, TP=4. Triggered on stock loss and high-impact exchanges; outputs grounded counterfactual JSON with frame-data citations.
- **Mac client:** UVC capture (Elgato HD60 X), local CV tier (damage OCR, character bbox), live AR HUD via OpenCV, post-event rewind card UI.

Quantitative counterfactual claims (frame advantage, punish windows) are computed by a deterministic rule-based scorer over scraped frame data — not VLM-vibed.

## Demo

Joker vs. Toon Link, Final Destination, 1 stock.

## Project layout

- `mac/` — capture, perception, HUD, control flow
- `server/` — vLLM + SGLang inference services
- `data/` — frame data, hitboxes, UI region calibration
- `docs/superpowers/specs/` — design doc
- `docs/superpowers/plans/` — implementation plan

## Running

```
uv sync --extra dev
uv run python -m mac.main
```
