# Smash Coach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a live AR coach + post-event counterfactual replay engine for Super Smash Bros. Ultimate, demoed at 24h Etched inference hackathon. Two-player demo: Joker vs. Toon Link on Final Destination, 1 stock.

**Architecture:** Mac client ingests HDMI via UVC capture, runs local Tier 0/1 CV at 60Hz (damage %, stocks, character bbox), and streams 5–10Hz JPEGs to two GPU services on a Prime Intellect 8×H100 node. System 1 (Qwen2.5-VL-7B on 2 GPUs via vLLM) returns structured per-player action labels and intent. System 2 (Qwen2.5-VL-72B AWQ on 4 GPUs via SGLang, plus 2 spare for stretch spec-decode) is triggered on stock loss + high-impact exchanges with a 5s state buffer + saliency keyframes; it returns structured counterfactual JSON. A local rule-based scorer derives quantitative claims (frame advantage, punish window) from a static frame-data JSON for the two characters. Mac renders the AR HUD via OpenCV and a Rewind Card UI for post-event analysis.

**Tech Stack:** Python 3.11, OpenCV, pydantic v2, websockets, httpx, FastAPI, vLLM, SGLang, Qwen2.5-VL-7B + Qwen2.5-VL-72B-AWQ, BeautifulSoup (frame data scrape), pytest.

**Spec:** `docs/superpowers/specs/2026-06-19-smash-coach-design.md`

**Team:** Two engineers. Tasks tagged `[A]`, `[B]`, or `[joint]`. Parallelizable tasks across phases marked at the top of each task.

---

## File Structure

```
repo/
├── pyproject.toml
├── mac/
│   ├── __init__.py
│   ├── capture.py              # UVC capture loop
│   ├── tier0_ocr.py            # Damage %, stocks, timer (template/CNN)
│   ├── tier1_cv.py             # Character bbox, position, velocity
│   ├── state.py                # s_t schema + rolling buffer
│   ├── dispatcher.py           # WebSocket client → System 1
│   ├── trigger.py              # Event detection for System 2
│   ├── system2_client.py       # HTTP client → System 2
│   ├── scorer.py               # Rule-based frame-data scorer
│   ├── hud.py                  # OpenCV overlay renderer
│   ├── rewind_card.py          # Rewind card UI (pygame surface)
│   ├── frame_data.py           # Frame-data + hitbox loader
│   └── main.py                 # Orchestrator
├── server/
│   ├── __init__.py
│   ├── system1_server.py       # FastAPI + vLLM
│   ├── system2_server.py       # FastAPI + SGLang
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── system1.py          # S1 system prompt + JSON schema
│   │   └── system2.py          # S2 system prompt + JSON schema
│   ├── launch_system1.sh
│   └── launch_system2.sh
├── data/
│   ├── frame_data.json
│   ├── hitboxes.json
│   └── ui_regions.json
├── scripts/
│   ├── scrape_frame_data.py
│   ├── smoke_capture.py
│   ├── smoke_system1.py
│   └── smoke_system2.py
└── tests/
    ├── test_state.py
    ├── test_trigger.py
    ├── test_scorer.py
    ├── test_frame_data.py
    └── test_prompts_schema.py
```

---

## Phase 0: Repo bootstrap (H0–H0:30)

### Task 0.1: Initialize repo [joint]

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `mac/__init__.py`, `server/__init__.py`, `server/prompts/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Initialize git and python project**

```bash
cd /Users/thondascully/Projects/etched
git init
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "smash-coach"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "opencv-python==4.10.0.84",
  "numpy>=1.26",
  "pydantic>=2.7",
  "websockets>=12.0",
  "httpx>=0.27",
  "fastapi>=0.110",
  "uvicorn>=0.29",
  "pygame>=2.5",
  "beautifulsoup4>=4.12",
  "requests>=2.32",
  "pytesseract>=0.3.10",
  "pillow>=10.3",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]
server = ["vllm>=0.6.0", "sglang>=0.3.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/raw/
*.mp4
*.jpg
.DS_Store
```

- [ ] **Step 4: Install and create empty `__init__.py` files**

```bash
pip install -e ".[dev]"
mkdir -p mac server/prompts tests data scripts
touch mac/__init__.py server/__init__.py server/prompts/__init__.py tests/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: initial repo scaffold"
```

---

## Phase 1: Data layer + state schema (H0:30–H2)

### Task 1.1: $s_t$ pydantic schema [A]

**Files:**
- Create: `mac/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_state.py
from mac.state import StateT, PlayerState, ActionState, Intent, Phase

def test_state_t_round_trip():
    s = StateT(
        t=1.23,
        damage={"p1": 45.0, "p2": 12.5},
        stocks={"p1": 1, "p2": 1},
        positions={
            "p1": PlayerState(x=100.0, y=200.0, facing="right",
                              airborne=False, vx=0.0, vy=0.0),
            "p2": PlayerState(x=300.0, y=200.0, facing="left",
                              airborne=True, vx=-1.5, vy=2.0),
        },
        actions={
            "p1": ActionState(label="neutral", phase="neutral",
                              confidence=0.9, onset_estimate_t=1.0),
            "p2": ActionState(label="joker_fsmash", phase="startup",
                              confidence=0.8, onset_estimate_t=1.1),
        },
        intent={"p1": "neutral", "p2": "pressuring"},
    )
    blob = s.model_dump_json()
    s2 = StateT.model_validate_json(blob)
    assert s2.actions["p2"].label == "joker_fsmash"
    assert s2.derived.distance is not None  # computed by validator
```

- [ ] **Step 2: Run test, expect FAIL**

```bash
pytest tests/test_state.py -v
```
Expected: `ImportError: cannot import name 'StateT'`.

- [ ] **Step 3: Implement `mac/state.py`**

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator
import math
import time
from collections import deque

Facing = Literal["left", "right"]
Phase = Literal["startup", "active", "endlag", "neutral", "unknown"]
Intent = Literal["pressuring", "ledge-trapping", "neutral",
                 "recovering", "punishing", "unknown"]

class PlayerState(BaseModel):
    x: float
    y: float
    facing: Facing
    airborne: bool
    vx: float = 0.0
    vy: float = 0.0

class ActionState(BaseModel):
    label: str  # e.g. "joker_fsmash"; closed vocab enforced by prompts
    phase: Phase = "unknown"
    confidence: float = 0.0
    onset_estimate_t: float = 0.0

class Derived(BaseModel):
    distance: float = 0.0
    relative_facing: Literal["facing", "back-turned", "mixed"] = "mixed"
    ledge_owner: Optional[Literal["p1", "p2"]] = None
    stage_control_estimate: float = 0.5  # 0=p2 dominates, 1=p1 dominates
    active_punish_window_for: Optional[Literal["p1", "p2"]] = None

class StateT(BaseModel):
    t: float
    damage: dict[str, float]
    stocks: dict[str, int]
    positions: dict[str, PlayerState]
    actions: dict[str, ActionState]
    intent: dict[str, Intent] = Field(
        default_factory=lambda: {"p1": "neutral", "p2": "neutral"})
    derived: Derived = Field(default_factory=Derived)
    controller_input_t: Optional[dict] = None  # reserved for future

    @model_validator(mode="after")
    def compute_derived(self) -> "StateT":
        p1, p2 = self.positions["p1"], self.positions["p2"]
        self.derived.distance = math.hypot(p1.x - p2.x, p1.y - p2.y)
        self.derived.relative_facing = (
            "facing" if (p1.facing == "right" and p1.x < p2.x)
                    or (p1.facing == "left" and p1.x > p2.x)
            else "back-turned"
        )
        return self
```

- [ ] **Step 4: Run test, expect PASS**

```bash
pytest tests/test_state.py -v
```

- [ ] **Step 5: Add a rolling buffer test**

```python
def test_state_buffer_window():
    from mac.state import StateBuffer
    buf = StateBuffer(window_seconds=2.0)
    now = time.time()
    for i in range(20):
        s = make_state(t=now + i * 0.2)  # helper at top of file
        buf.push(s)
    window = buf.window(now + 4.0 - 2.0, now + 4.0)
    assert all(s.t >= (now + 2.0) for s in window)
```

Add helper at the top of `tests/test_state.py`:

```python
def make_state(t):
    return StateT(
        t=t, damage={"p1": 0.0, "p2": 0.0}, stocks={"p1": 1, "p2": 1},
        positions={
            "p1": PlayerState(x=0, y=0, facing="right", airborne=False),
            "p2": PlayerState(x=100, y=0, facing="left", airborne=False),
        },
        actions={
            "p1": ActionState(label="neutral"),
            "p2": ActionState(label="neutral"),
        },
    )
```

- [ ] **Step 6: Implement `StateBuffer`**

Append to `mac/state.py`:

```python
class StateBuffer:
    def __init__(self, window_seconds: float = 10.0):
        self.window_seconds = window_seconds
        self._buf: deque[StateT] = deque()

    def push(self, s: StateT) -> None:
        self._buf.append(s)
        cutoff = s.t - self.window_seconds
        while self._buf and self._buf[0].t < cutoff:
            self._buf.popleft()

    def latest(self) -> Optional[StateT]:
        return self._buf[-1] if self._buf else None

    def window(self, t_start: float, t_end: float) -> list[StateT]:
        return [s for s in self._buf if t_start <= s.t <= t_end]
```

- [ ] **Step 7: Run all tests, commit**

```bash
pytest tests/test_state.py -v
git add mac/state.py tests/test_state.py
git commit -m "feat(state): pydantic s_t schema + rolling buffer"
```

---

### Task 1.2: Frame data scraper [B] (parallel with 1.1)

**Files:**
- Create: `scripts/scrape_frame_data.py`
- Create: `data/frame_data.json` (output)

- [ ] **Step 1: Write scraper**

```python
# scripts/scrape_frame_data.py
"""
Scrape ultimateframedata.com for Joker and Toon Link move data.
Writes to data/frame_data.json with schema:
{
  "joker": [
    {"name": "jab1", "startup_f": 4, "active_f": [4,5], "endlag_f": 13,
     "landing_lag_f": null, "shield_advantage": -8, "on_hit_advantage": null,
     "range_estimate": "short", "category": "ground"},
    ...
  ],
  "toon_link": [...]
}
"""
import json, re, sys, time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

URLS = {
    "joker": "https://ultimateframedata.com/joker",
    "toon_link": "https://ultimateframedata.com/toon_link",
}

def parse_active(text: str):
    m = re.search(r"(\d+)(?:-(\d+))?", text)
    if not m: return None
    a = int(m.group(1)); b = int(m.group(2) or a)
    return [a, b]

def parse_move_block(block) -> dict | None:
    name_el = block.find(class_="movename")
    if not name_el: return None
    name = name_el.get_text(strip=True).lower().replace(" ", "_")
    def field(label):
        row = block.find(string=re.compile(label, re.I))
        return row.parent.find_next("td").get_text(strip=True) if row else ""
    startup = field("Startup")
    active = field("Hitbox|Active")
    endlag = field("FAF|Endlag|Total")
    shield_adv = field("Shield Adv|Shield advantage")
    return {
        "name": name,
        "startup_f": int(re.search(r"\d+", startup).group()) if re.search(r"\d+", startup) else None,
        "active_f": parse_active(active),
        "endlag_f": int(re.search(r"\d+", endlag).group()) if re.search(r"\d+", endlag) else None,
        "landing_lag_f": None,
        "shield_advantage": int(re.search(r"-?\d+", shield_adv).group()) if re.search(r"-?\d+", shield_adv) else None,
        "on_hit_advantage": None,
        "range_estimate": "medium",  # filled in manually later
        "category": "ground" if "smash" in name or "jab" in name else "aerial" if "air" in name else "special",
    }

def scrape(char: str, url: str) -> list[dict]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    blocks = soup.find_all(class_="move")
    moves = []
    for b in blocks:
        m = parse_move_block(b)
        if m and m["startup_f"] is not None:
            moves.append(m)
    return moves

def main():
    out = {}
    for char, url in URLS.items():
        print(f"scraping {char}...", file=sys.stderr)
        out[char] = scrape(char, url)
        time.sleep(1)
    Path("data/frame_data.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {sum(len(v) for v in out.values())} moves")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run scraper**

```bash
python scripts/scrape_frame_data.py
```
Expected: writes `data/frame_data.json`, reports >40 moves.

- [ ] **Step 3: Spot-check output**

```bash
python -c "import json; d=json.load(open('data/frame_data.json')); print(len(d['joker']), len(d['toon_link'])); print(d['joker'][:2])"
```
Expected: ~20–35 moves per char, structured JSON.

- [ ] **Step 4: If scrape parser misses fields, hand-patch manually**

Open `data/frame_data.json`, fix any `null` startups or active ranges for the core ~12 moves per character that will appear in demo play (jab, ftilt, utilt, dtilt, fsmash, usmash, dsmash, nair, fair, bair, uair, dair, neutral B, side B, up B, down B, grab). Spend max 30 min — if a move's data is missing, mark as `"active_f": [99, 99]` so scorer treats it as never-active rather than crashing.

- [ ] **Step 5: Commit**

```bash
git add scripts/scrape_frame_data.py data/frame_data.json
git commit -m "feat(data): scrape Joker + Toon Link frame data"
```

---

### Task 1.3: Frame data loader + hitbox stubs [B]

**Files:**
- Create: `mac/frame_data.py`
- Create: `data/hitboxes.json`
- Test: `tests/test_frame_data.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_frame_data.py
from mac.frame_data import FrameData

def test_lookup_known_move():
    fd = FrameData.load("data/frame_data.json", "data/hitboxes.json")
    m = fd.move("joker", "fsmash")
    assert m.startup_f is not None
    assert m.startup_f > 0

def test_hitbox_circles_for_active_frame():
    fd = FrameData.load("data/frame_data.json", "data/hitboxes.json")
    circles = fd.hitboxes("joker", "fsmash", frame_in_move=18)
    assert isinstance(circles, list)  # may be empty if not active or no data
```

- [ ] **Step 2: Write `mac/frame_data.py`**

```python
from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path

class Move(BaseModel):
    name: str
    startup_f: Optional[int] = None
    active_f: Optional[list[int]] = None  # [first, last]
    endlag_f: Optional[int] = None
    landing_lag_f: Optional[int] = None
    shield_advantage: Optional[int] = None
    on_hit_advantage: Optional[int] = None
    range_estimate: str = "medium"
    category: str = "ground"

class HitboxCircle(BaseModel):
    dx: float
    dy: float
    radius: float
    active_frames: list[int]  # [first, last]

class FrameData:
    def __init__(self, moves: dict[str, dict[str, Move]],
                 hitboxes: dict[str, dict[str, list[HitboxCircle]]]):
        self.moves = moves
        self.hitboxes_data = hitboxes

    @classmethod
    def load(cls, frame_data_path: str, hitboxes_path: str) -> "FrameData":
        raw = json.loads(Path(frame_data_path).read_text())
        moves = {
            char: {m["name"]: Move(**m) for m in lst}
            for char, lst in raw.items()
        }
        hb_raw = (json.loads(Path(hitboxes_path).read_text())
                  if Path(hitboxes_path).exists() else {})
        hitboxes = {
            char: {mv: [HitboxCircle(**c) for c in circles]
                   for mv, circles in mv_dict.items()}
            for char, mv_dict in hb_raw.items()
        }
        return cls(moves, hitboxes)

    def move(self, char: str, name: str) -> Move:
        return self.moves[char][name]

    def hitboxes(self, char: str, name: str, frame_in_move: int) -> list[HitboxCircle]:
        char_hb = self.hitboxes_data.get(char, {})
        circles = char_hb.get(name, [])
        return [c for c in circles
                if c.active_frames[0] <= frame_in_move <= c.active_frames[1]]
```

- [ ] **Step 3: Write `data/hitboxes.json` minimal stub**

```json
{
  "joker": {
    "fsmash": [
      {"dx": 35, "dy": 0, "radius": 18, "active_frames": [16, 19]}
    ],
    "usmash": [
      {"dx": 0, "dy": -30, "radius": 22, "active_frames": [14, 17]}
    ]
  },
  "toon_link": {
    "fsmash": [
      {"dx": 30, "dy": 0, "radius": 16, "active_frames": [15, 17]}
    ]
  }
}
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_frame_data.py -v
```

- [ ] **Step 5: Commit**

```bash
git add mac/frame_data.py data/hitboxes.json tests/test_frame_data.py
git commit -m "feat(data): frame data + hitbox loader"
```

---

## Phase 2: GPU node bring-up (H0:30–H2, parallel with Phase 1)

### Task 2.1: System 1 server (vLLM, Qwen2.5-VL-7B) [B]

**Files:**
- Create: `server/system1_server.py`
- Create: `server/prompts/system1.py`
- Create: `server/launch_system1.sh`

- [ ] **Step 1: SSH to Prime Intellect node, install vllm**

```bash
ssh <node>
nvidia-smi  # confirm 8x H100s visible
python3 -m venv /workspace/.venv
source /workspace/.venv/bin/activate
pip install vllm==0.6.3 fastapi uvicorn pydantic httpx
```

- [ ] **Step 2: Write `server/prompts/system1.py`**

```python
JOKER_ACTIONS = [
    "neutral", "walk", "dash", "shield", "roll", "spotdodge", "airdodge",
    "jab", "ftilt", "utilt", "dtilt", "fsmash", "usmash", "dsmash",
    "nair", "fair", "bair", "uair", "dair", "grab",
    "eiha", "eigaon", "tetrakarn", "makarakarn",
    "gun", "arsene_summon", "wings_of_rebellion",
]
TOONLINK_ACTIONS = [
    "neutral", "walk", "dash", "shield", "roll", "spotdodge", "airdodge",
    "jab", "ftilt", "utilt", "dtilt", "fsmash", "usmash", "dsmash",
    "nair", "fair", "bair", "uair", "dair", "grab",
    "boomerang", "bomb_pull", "bomb_throw", "arrow", "spin_attack",
    "hookshot",
]
PHASES = ["startup", "active", "endlag", "neutral", "unknown"]
INTENTS = ["pressuring", "ledge-trapping", "neutral",
           "recovering", "punishing", "unknown"]

SYSTEM_PROMPT = """You are a Smash Ultimate frame analyst.
Given a single game frame (Joker vs Toon Link on Final Destination), output STRICT JSON:

{
  "p1": {
    "action_label": <one of joker actions>,
    "phase": <one of phases>,
    "confidence": <0..1>,
    "intent": <one of intents>
  },
  "p2": {
    "action_label": <one of toon link actions>,
    "phase": <one of phases>,
    "confidence": <0..1>,
    "intent": <one of intents>
  }
}

p1 = Joker (left side at match start). p2 = Toon Link.
If unsure, use "unknown" actions with confidence < 0.3.
Output ONLY the JSON, no prose.
"""

def build_json_schema():
    return {
        "type": "object",
        "properties": {
            "p1": _player_schema(JOKER_ACTIONS),
            "p2": _player_schema(TOONLINK_ACTIONS),
        },
        "required": ["p1", "p2"],
    }

def _player_schema(actions):
    return {
        "type": "object",
        "properties": {
            "action_label": {"type": "string", "enum": actions},
            "phase": {"type": "string", "enum": PHASES},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "intent": {"type": "string", "enum": INTENTS},
        },
        "required": ["action_label", "phase", "confidence", "intent"],
    }
```

- [ ] **Step 3: Write `server/system1_server.py`**

```python
import base64
import io
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json

from server.prompts.system1 import SYSTEM_PROMPT, build_json_schema

app = FastAPI()
llm: LLM | None = None
SCHEMA = build_json_schema()

class FrameIn(BaseModel):
    image_b64: str  # JPEG base64
    t: float

@app.on_event("startup")
def load_model():
    global llm
    llm = LLM(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
    )

@app.post("/infer")
def infer(req: FrameIn):
    img = Image.open(io.BytesIO(base64.b64decode(req.image_b64)))
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        guided_decoding=GuidedDecodingParams(json=SCHEMA),
    )
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Analyze this frame."},
        ]},
    ]
    out = llm.chat(prompt, sampling_params=sampling)
    return json.loads(out[0].outputs[0].text)
```

- [ ] **Step 4: Write `server/launch_system1.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1
exec uvicorn server.system1_server:app --host 0.0.0.0 --port 8001
```

- [ ] **Step 5: Launch + smoke-test from Mac**

```bash
# on node:
chmod +x server/launch_system1.sh
./server/launch_system1.sh &
# on Mac:
curl -X POST http://<node>:8001/infer \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\":\"$(base64 < tests/fixtures/sample_frame.jpg)\", \"t\": 0.0}"
```
Expected: JSON with `p1`/`p2` keys conforming to schema. First call slow (~30s warmup), subsequent calls <500ms.

- [ ] **Step 6: Commit**

```bash
git add server/system1_server.py server/prompts/system1.py server/launch_system1.sh
git commit -m "feat(server): system1 vLLM Qwen2.5-VL-7B endpoint"
```

---

### Task 2.2: System 2 server (SGLang, Qwen2.5-VL-72B-AWQ) [B]

**Files:**
- Create: `server/system2_server.py`
- Create: `server/prompts/system2.py`
- Create: `server/launch_system2.sh`

- [ ] **Step 1: Install SGLang on node**

```bash
ssh <node>
source /workspace/.venv/bin/activate
pip install "sglang[all]==0.3.6"
```

- [ ] **Step 2: Write `server/prompts/system2.py`**

```python
COUNTERFACTUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": 400},
        "chosen_action": {
            "type": "object",
            "properties": {
                "player": {"type": "string", "enum": ["p1", "p2"]},
                "action_label": {"type": "string"},
                "frame_t": {"type": "number"},
                "reasoning": {"type": "string", "maxLength": 300},
            },
            "required": ["player", "action_label", "frame_t", "reasoning"],
        },
        "counterfactual_action": {
            "type": "object",
            "properties": {
                "action_label": {"type": "string"},
                "rationale": {"type": "string", "maxLength": 300},
                "qualitative_likelihood": {
                    "type": "string",
                    "enum": ["likely", "plausible", "speculative"],
                },
            },
            "required": ["action_label", "rationale", "qualitative_likelihood"],
        },
        "frame_data_citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "character": {"type": "string"},
                    "move": {"type": "string"},
                    "stat": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["character", "move", "stat", "value"],
            },
            "maxItems": 6,
        },
    },
    "required": ["summary", "chosen_action", "counterfactual_action",
                 "frame_data_citations"],
}

def build_system_prompt(frame_data_blob: str) -> str:
    return f"""You are a Smash Ultimate frame-data analyst.
You analyze a 5-second exchange and produce a counterfactual analysis.

You will receive:
- Frame data table (authoritative, below)
- A timestamped state trajectory (s_t sequence)
- Saliency keyframes (images)

Output STRICT JSON per schema. Rules:
- chosen_action.action_label MUST appear in the frame data table.
- counterfactual_action.action_label MUST appear in the frame data table.
- Cite real moves only. Never invent move names or numbers.
- Use ONLY qualitative likelihood ("likely"/"plausible"/"speculative") — never a percentage.

FRAME DATA TABLE:
{frame_data_blob}
"""
```

- [ ] **Step 3: Write `server/system2_server.py`**

```python
import base64, io, json
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import sglang as sgl

from server.prompts.system2 import COUNTERFACTUAL_SCHEMA, build_system_prompt

app = FastAPI()
runtime: sgl.Runtime | None = None
SYS_PROMPT: str | None = None

class Keyframe(BaseModel):
    image_b64: str
    t: float

class CounterfactualReq(BaseModel):
    state_trajectory: list[dict]
    keyframes: list[Keyframe]
    event_type: str  # "stock_loss" | "exchange"

@app.on_event("startup")
def load_model():
    global runtime, SYS_PROMPT
    runtime = sgl.Runtime(
        model_path="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        tp_size=4,
        mem_fraction_static=0.85,
    )
    sgl.set_default_backend(runtime)
    frame_data = Path("data/frame_data.json").read_text()
    SYS_PROMPT = build_system_prompt(frame_data)

@sgl.function
def analyze(s, system_prompt, user_text, images):
    s += sgl.system(system_prompt)
    s += sgl.user(sgl.image(images[0]) if images else "")
    for img in images[1:]:
        s += sgl.image(img)
    s += sgl.user(user_text)
    s += sgl.assistant(sgl.gen("out", max_tokens=800,
                               temperature=0.2,
                               json_schema=json.dumps(COUNTERFACTUAL_SCHEMA)))

@app.post("/counterfactual")
def counterfactual(req: CounterfactualReq):
    images = [Image.open(io.BytesIO(base64.b64decode(kf.image_b64)))
              for kf in req.keyframes]
    user_text = (
        f"Event: {req.event_type}\n"
        f"State trajectory ({len(req.state_trajectory)} steps):\n"
        f"{json.dumps(req.state_trajectory, indent=2)}\n\n"
        f"Analyze this exchange and produce the counterfactual JSON."
    )
    state = analyze.run(system_prompt=SYS_PROMPT,
                        user_text=user_text, images=images)
    return json.loads(state["out"])
```

- [ ] **Step 4: Write `server/launch_system2.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2,3,4,5
exec uvicorn server.system2_server:app --host 0.0.0.0 --port 8002
```

- [ ] **Step 5: Smoke-test from Mac**

Create `scripts/smoke_system2.py`:

```python
import base64, json, httpx
from pathlib import Path

img = base64.b64encode(Path("tests/fixtures/sample_frame.jpg").read_bytes()).decode()
trajectory = [
    {"t": 0.0, "damage": {"p1": 30, "p2": 40},
     "actions": {"p1": {"label": "shield"}, "p2": {"label": "fsmash"}}},
    {"t": 0.2, "damage": {"p1": 45, "p2": 40},
     "actions": {"p1": {"label": "neutral"}, "p2": {"label": "neutral"}}},
]
r = httpx.post("http://<node>:8002/counterfactual", json={
    "state_trajectory": trajectory,
    "keyframes": [{"image_b64": img, "t": 0.1}],
    "event_type": "exchange",
}, timeout=60.0)
print(json.dumps(r.json(), indent=2))
```

Run:
```bash
python scripts/smoke_system2.py
```
Expected: returns JSON conforming to `COUNTERFACTUAL_SCHEMA`. First call slow (~60s warmup).

- [ ] **Step 6: Commit**

```bash
git add server/system2_server.py server/prompts/system2.py server/launch_system2.sh scripts/smoke_system2.py
git commit -m "feat(server): system2 SGLang Qwen2.5-VL-72B counterfactual endpoint"
```

---

## Phase 3: Mac client foundation (H2–H6)

### Task 3.1: Capture loop [A]

**Files:**
- Create: `mac/capture.py`
- Create: `scripts/smoke_capture.py`

- [ ] **Step 1: Implement `mac/capture.py`**

```python
import cv2
import time
from dataclasses import dataclass
from typing import Iterator

@dataclass
class Frame:
    img: "any"   # np.ndarray BGR
    t: float     # monotonic timestamp

class Capture:
    def __init__(self, device_index: int = 0, width: int = 1920, height: int = 1080):
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        if not self.cap.isOpened():
            raise RuntimeError("Capture device not opened")

    def frames(self) -> Iterator[Frame]:
        while True:
            ok, img = self.cap.read()
            if not ok:
                continue
            yield Frame(img=img, t=time.monotonic())

    def close(self):
        self.cap.release()
```

- [ ] **Step 2: Write smoke script**

```python
# scripts/smoke_capture.py
import cv2
from mac.capture import Capture

cap = Capture(device_index=0)
for i, frame in enumerate(cap.frames()):
    cv2.imshow("capture", frame.img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.close()
cv2.destroyAllWindows()
```

- [ ] **Step 3: Run, verify Switch screen is visible**

```bash
python scripts/smoke_capture.py
```
Expected: window shows live Switch output. If wrong device index, try 1/2 etc.

- [ ] **Step 4: Save a sample frame for tests**

```bash
mkdir -p tests/fixtures
python -c "from mac.capture import Capture; import cv2; c=Capture(0); f=next(c.frames()); cv2.imwrite('tests/fixtures/sample_frame.jpg', f.img); c.close()"
```

- [ ] **Step 5: Commit**

```bash
git add mac/capture.py scripts/smoke_capture.py tests/fixtures/sample_frame.jpg
git commit -m "feat(mac): capture loop via UVC"
```

---

### Task 3.2: UI region calibration + Tier 0 OCR [A]

**Files:**
- Create: `mac/tier0_ocr.py`
- Create: `data/ui_regions.json`
- Create: `scripts/calibrate_ui.py`

- [ ] **Step 1: Hand-calibrate UI regions on FD**

Run a Smash match on FD, paused with a clean view. Take a screenshot via the capture loop. Open in Preview / any image editor. Note pixel rectangles for:
- P1 damage % crop
- P2 damage % crop
- P1 stock icons crop
- P2 stock icons crop
- Match timer crop

Write `data/ui_regions.json`:

```json
{
  "resolution": [1920, 1080],
  "p1_damage": [430, 880, 200, 110],
  "p2_damage": [1290, 880, 200, 110],
  "p1_stocks": [430, 1000, 150, 40],
  "p2_stocks": [1290, 1000, 150, 40],
  "timer": [870, 60, 180, 80]
}
```
(Format: `[x, y, w, h]`. Adjust based on actual capture frames.)

- [ ] **Step 2: Implement Tier 0 OCR**

```python
# mac/tier0_ocr.py
import cv2
import json
import re
from pathlib import Path
import pytesseract
import numpy as np

class Tier0:
    def __init__(self, regions_path: str = "data/ui_regions.json"):
        cfg = json.loads(Path(regions_path).read_text())
        self.regions = cfg
        self.base_res = tuple(cfg["resolution"])

    def _crop(self, img, key: str):
        h, w = img.shape[:2]
        sx, sy = w / self.base_res[0], h / self.base_res[1]
        x, y, ww, hh = self.regions[key]
        x, y, ww, hh = int(x*sx), int(y*sy), int(ww*sx), int(hh*sy)
        return img[y:y+hh, x:x+ww]

    def damage(self, img, who: str) -> float | None:
        crop = self._crop(img, f"{who}_damage")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(
            th, config="--psm 7 -c tessedit_char_whitelist=0123456789.%")
        m = re.search(r"\d+(?:\.\d+)?", text)
        return float(m.group()) if m else None

    def stocks(self, img, who: str) -> int:
        crop = self._crop(img, f"{who}_stocks")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        # rough heuristic: count distinct blobs
        n_labels, _ = cv2.connectedComponents(th)
        return max(0, min(3, n_labels - 1))
```

- [ ] **Step 3: Manual smoke-test**

```python
# scripts/smoke_ocr.py
from mac.capture import Capture
from mac.tier0_ocr import Tier0

cap = Capture(0); ocr = Tier0()
for frame in cap.frames():
    d1 = ocr.damage(frame.img, "p1")
    d2 = ocr.damage(frame.img, "p2")
    s1 = ocr.stocks(frame.img, "p1")
    print(f"p1: {d1}% ({s1} stocks)  p2: {d2}%", end="\r")
```

Expected: damage updates as players take hits. If garbled, tune the OCR threshold and crop region. **Time-box to 60 minutes.** If OCR is unreliable, fall back to a small template-match per digit (the Smash damage font is fixed).

- [ ] **Step 4: Commit**

```bash
git add mac/tier0_ocr.py data/ui_regions.json scripts/smoke_ocr.py
git commit -m "feat(mac): tier 0 OCR for damage and stocks"
```

---

### Task 3.3: Tier 1 character bounding boxes [A]

**Files:**
- Create: `mac/tier1_cv.py`

**Approach:** No model training in 24h. Use HSV color masking on Joker (red/black palette) and Toon Link (green tunic) plus connected-components blob detection. This is brittle for fighting effects but workable on FD.

- [ ] **Step 1: Implement HSV-based bbox detection**

```python
# mac/tier1_cv.py
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class Bbox:
    x: float; y: float; w: float; h: float; cx: float; cy: float

# tuned HSV ranges (adjust empirically)
JOKER_HSV = [(0, 80, 80), (10, 255, 255)]      # red coat
TOONLINK_HSV = [(40, 80, 80), (80, 255, 255)]  # green tunic

def _bbox_from_mask(mask) -> Bbox | None:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 400: return None
    x, y, w, h = cv2.boundingRect(c)
    return Bbox(float(x), float(y), float(w), float(h),
                float(x + w/2), float(y + h/2))

def detect(img) -> dict[str, Bbox | None]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = {}
    for name, (lo, hi) in [("p1", JOKER_HSV), ("p2", TOONLINK_HSV)]:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        out[name] = _bbox_from_mask(mask)
    return out
```

- [ ] **Step 2: Smoke test (visual)**

```python
# scripts/smoke_cv.py
import cv2
from mac.capture import Capture
from mac.tier1_cv import detect

cap = Capture(0)
for f in cap.frames():
    bboxes = detect(f.img)
    for name, b in bboxes.items():
        if b:
            cv2.rectangle(f.img, (int(b.x), int(b.y)),
                          (int(b.x+b.w), int(b.y+b.h)), (0,255,0), 2)
            cv2.putText(f.img, name, (int(b.x), int(b.y)-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imshow("cv", f.img)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
```

Run and tune HSV ranges by watching the live boxes. **Time-box: 45 min.** If unreliable, accept noisy boxes — System 1 will smooth via action labels.

- [ ] **Step 3: Commit**

```bash
git add mac/tier1_cv.py scripts/smoke_cv.py
git commit -m "feat(mac): tier 1 HSV-based character bbox"
```

---

## Phase 4: System 1 wiring (H6–H10)

### Task 4.1: WebSocket dispatcher → System 1 [A]

**Files:**
- Create: `mac/dispatcher.py`

- [ ] **Step 1: Implement dispatcher**

```python
# mac/dispatcher.py
import asyncio
import base64
import cv2
import httpx
import time
from dataclasses import dataclass
from typing import Callable

@dataclass
class S1Out:
    p1: dict; p2: dict; t: float

class System1Client:
    def __init__(self, url: str = "http://node:8001/infer", hz: float = 7.0):
        self.url = url
        self.min_interval = 1.0 / hz
        self._last_sent = 0.0
        self._client = httpx.AsyncClient(timeout=2.0)

    async def maybe_infer(self, img, t: float) -> S1Out | None:
        now = time.monotonic()
        if now - self._last_sent < self.min_interval:
            return None
        self._last_sent = now
        small = cv2.resize(img, (640, 640))
        _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(jpg.tobytes()).decode()
        try:
            r = await self._client.post(self.url,
                                        json={"image_b64": b64, "t": t})
            r.raise_for_status()
            d = r.json()
            return S1Out(p1=d["p1"], p2=d["p2"], t=t)
        except (httpx.HTTPError, KeyError):
            return None
```

(Note: started as WebSocket in spec but HTTP is simpler and the per-frame overhead is negligible at 7Hz over a wired link. Schema accepts a swap to WS later.)

- [ ] **Step 2: Commit**

```bash
git add mac/dispatcher.py
git commit -m "feat(mac): system1 HTTP dispatcher with rate limit"
```

---

### Task 4.2: Onset estimator [A]

**Files:**
- Create: `mac/onset.py`
- Test: `tests/test_onset.py`

**Logic:** When an action label transitions from anything → a non-neutral action, record `onset_estimate_t`. This is ±100–200ms accurate (set by System 1 polling rate). Hitbox rendering uses this + `frame_in_move = (now - onset_estimate_t) * 60`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_onset.py
from mac.onset import OnsetTracker

def test_onset_records_transition():
    tr = OnsetTracker()
    tr.update("p1", "neutral", 1.0)
    tr.update("p1", "neutral", 1.1)
    tr.update("p1", "fsmash", 1.2)
    assert tr.onset("p1") == 1.2

def test_onset_persists_during_action():
    tr = OnsetTracker()
    tr.update("p1", "fsmash", 1.0)
    tr.update("p1", "fsmash", 1.2)
    assert tr.onset("p1") == 1.0

def test_onset_resets_on_new_action():
    tr = OnsetTracker()
    tr.update("p1", "fsmash", 1.0)
    tr.update("p1", "shield", 1.5)
    assert tr.onset("p1") == 1.5
```

- [ ] **Step 2: Implement**

```python
# mac/onset.py
class OnsetTracker:
    def __init__(self):
        self._last_action: dict[str, str] = {}
        self._onset_t: dict[str, float] = {}

    def update(self, player: str, action: str, t: float) -> None:
        if self._last_action.get(player) != action:
            self._onset_t[player] = t
            self._last_action[player] = action

    def onset(self, player: str) -> float | None:
        return self._onset_t.get(player)

    def frame_in_move(self, player: str, now: float) -> int:
        o = self._onset_t.get(player)
        if o is None: return 0
        return int(max(0, (now - o) * 60))
```

- [ ] **Step 3: Run tests, commit**

```bash
pytest tests/test_onset.py -v
git add mac/onset.py tests/test_onset.py
git commit -m "feat(mac): action onset tracker"
```

---

### Task 4.3: HUD renderer [A]

**Files:**
- Create: `mac/hud.py`

- [ ] **Step 1: Implement HUD renderer**

```python
# mac/hud.py
import cv2
import math
from mac.state import StateT
from mac.frame_data import FrameData
from mac.onset import OnsetTracker

class HUD:
    def __init__(self, fd: FrameData, onset: OnsetTracker):
        self.fd = fd; self.onset = onset

    def draw(self, img, s: StateT, now: float, char_map: dict[str, str]):
        out = img.copy()
        self._draw_damage(out, s)
        self._draw_action_labels(out, s)
        self._draw_hitboxes(out, s, now, char_map)
        self._draw_threat_zones(out, s, char_map)
        return out

    def _draw_damage(self, img, s):
        h, w = img.shape[:2]
        cv2.putText(img, f"P1 {s.damage['p1']:.1f}%",
                    (40, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
        cv2.putText(img, f"P2 {s.damage['p2']:.1f}%",
                    (w-340, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

    def _draw_action_labels(self, img, s):
        for who, color in [("p1", (255,200,0)), ("p2", (0,200,255))]:
            p = s.positions[who]; a = s.actions[who]
            txt = f"{a.label} [{a.phase}]"
            cv2.putText(img, txt, (int(p.x), int(p.y)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_hitboxes(self, img, s, now, char_map):
        for who in ("p1", "p2"):
            char = char_map[who]
            mv = s.actions[who].label
            try:
                self.fd.move(char, mv)
            except KeyError:
                continue
            fim = self.onset.frame_in_move(who, now)
            circles = self.fd.hitboxes(char, mv, fim)
            p = s.positions[who]
            sign = 1 if p.facing == "right" else -1
            for c in circles:
                cx, cy = int(p.x + sign*c.dx), int(p.y + c.dy)
                cv2.circle(img, (cx, cy), int(c.radius), (0, 0, 255), 2)

    def _draw_threat_zones(self, img, s, char_map):
        # cheap radius from move range_estimate
        for who, color in [("p1", (0, 100, 255)), ("p2", (100, 0, 255))]:
            char = char_map[who]
            mv = s.actions[who].label
            try:
                m = self.fd.move(char, mv)
            except KeyError:
                continue
            r = {"short": 40, "medium": 70, "long": 110}.get(m.range_estimate, 60)
            p = s.positions[who]
            cv2.circle(img, (int(p.x), int(p.y)), r, color, 1)
```

- [ ] **Step 2: Smoke test (visual)**

Create a quick integration script that builds a fake `StateT` from `Tier0 + Tier1 + a hardcoded action`, runs `HUD.draw` over each frame, and shows the result. Skip if you'd rather wire end-to-end in Phase 5.

- [ ] **Step 3: Commit**

```bash
git add mac/hud.py
git commit -m "feat(mac): HUD renderer (damage, action label, hitbox, threat zone)"
```

---

### Task 4.4: Main orchestrator [joint]

**Files:**
- Create: `mac/main.py`

- [ ] **Step 1: Implement main loop**

```python
# mac/main.py
import asyncio
import cv2
import time
import os

from mac.capture import Capture
from mac.tier0_ocr import Tier0
from mac.tier1_cv import detect as detect_bboxes
from mac.dispatcher import System1Client, S1Out
from mac.onset import OnsetTracker
from mac.state import StateT, PlayerState, ActionState, StateBuffer
from mac.frame_data import FrameData
from mac.hud import HUD

CHAR_MAP = {"p1": "joker", "p2": "toon_link"}

async def run():
    cap = Capture(device_index=int(os.getenv("CAP_DEV", "0")))
    ocr = Tier0()
    fd = FrameData.load("data/frame_data.json", "data/hitboxes.json")
    onset = OnsetTracker()
    hud = HUD(fd, onset)
    buf = StateBuffer(window_seconds=10.0)
    s1 = System1Client(url=os.getenv("S1_URL", "http://localhost:8001/infer"),
                       hz=float(os.getenv("S1_HZ", "7.0")))
    last_s1: S1Out | None = None

    for frame in cap.frames():
        t = frame.t
        # Tier 0
        d1 = ocr.damage(frame.img, "p1") or 0.0
        d2 = ocr.damage(frame.img, "p2") or 0.0
        st1 = ocr.stocks(frame.img, "p1")
        st2 = ocr.stocks(frame.img, "p2")
        # Tier 1
        bb = detect_bboxes(frame.img)
        def pos(b, default_x):
            if b is None:
                return PlayerState(x=default_x, y=500, facing="right",
                                   airborne=False)
            return PlayerState(x=b.cx, y=b.cy, facing="right",
                               airborne=False, vx=0, vy=0)
        positions = {
            "p1": pos(bb["p1"], 500.0),
            "p2": pos(bb["p2"], 1400.0),
        }
        # System 1 (async fire-and-forget)
        out = await s1.maybe_infer(frame.img, t)
        if out is not None:
            last_s1 = out
            onset.update("p1", out.p1["action_label"], t)
            onset.update("p2", out.p2["action_label"], t)
        actions = (
            {who: ActionState(label=getattr(last_s1, who)["action_label"],
                              phase=getattr(last_s1, who)["phase"],
                              confidence=getattr(last_s1, who)["confidence"],
                              onset_estimate_t=onset.onset(who) or t)
             for who in ("p1", "p2")}
            if last_s1 else
            {who: ActionState(label="neutral") for who in ("p1", "p2")}
        )
        intent = (
            {who: getattr(last_s1, who)["intent"] for who in ("p1", "p2")}
            if last_s1 else {"p1": "neutral", "p2": "neutral"}
        )
        s = StateT(
            t=t, damage={"p1": d1, "p2": d2}, stocks={"p1": st1, "p2": st2},
            positions=positions, actions=actions, intent=intent,
        )
        buf.push(s)
        composed = hud.draw(frame.img, s, t, CHAR_MAP)
        cv2.imshow("smash-coach", composed)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    asyncio.run(run())
```

- [ ] **Step 2: Run end-to-end**

```bash
S1_URL=http://<node>:8001/infer python -m mac.main
```
Expected: live HUD overlay with damage, action labels, hitboxes when smashes connect. Action labels lag ~150ms (System 1 polling) — acceptable.

- [ ] **Step 3: Commit (cut line H12 — must be working by here)**

```bash
git add mac/main.py
git commit -m "feat(mac): end-to-end main loop (tier0 + tier1 + s1 + hud)"
```

---

## Phase 5: System 2 wiring (H12–H18)

### Task 5.1: Event triggers [A]

**Files:**
- Create: `mac/trigger.py`
- Test: `tests/test_trigger.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_trigger.py
from mac.trigger import TriggerDetector, TriggerEvent
from tests.test_state import make_state  # reuse helper

def test_stock_loss_fires():
    td = TriggerDetector()
    s1 = make_state(0.0); s1.stocks = {"p1": 1, "p2": 1}
    s2 = make_state(0.5); s2.stocks = {"p1": 0, "p2": 1}
    assert td.check(s1) is None
    ev = td.check(s2)
    assert ev is not None and ev.kind == "stock_loss" and ev.who == "p1"

def test_damage_spike_fires():
    td = TriggerDetector(damage_delta=30.0, damage_window_s=2.0)
    s1 = make_state(0.0); s1.damage = {"p1": 0, "p2": 0}
    s2 = make_state(1.0); s2.damage = {"p1": 35, "p2": 0}
    td.check(s1)
    ev = td.check(s2)
    assert ev is not None and ev.kind == "exchange"

def test_cooldown_prevents_double_fire():
    td = TriggerDetector(damage_delta=30.0, cooldown_s=5.0)
    s1 = make_state(0.0); s1.damage = {"p1": 0, "p2": 0}
    s2 = make_state(1.0); s2.damage = {"p1": 35, "p2": 0}
    s3 = make_state(1.5); s3.damage = {"p1": 70, "p2": 0}
    td.check(s1); ev1 = td.check(s2); ev2 = td.check(s3)
    assert ev1 is not None and ev2 is None
```

- [ ] **Step 2: Implement**

```python
# mac/trigger.py
from dataclasses import dataclass
from typing import Optional
from mac.state import StateT

@dataclass
class TriggerEvent:
    kind: str   # "stock_loss" | "exchange"
    who: Optional[str]
    t: float

class TriggerDetector:
    def __init__(self,
                 damage_delta: float = 30.0,
                 damage_window_s: float = 2.0,
                 cooldown_s: float = 5.0):
        self.damage_delta = damage_delta
        self.damage_window_s = damage_window_s
        self.cooldown_s = cooldown_s
        self._history: list[StateT] = []
        self._last_fire_t: float = -1e9

    def check(self, s: StateT) -> Optional[TriggerEvent]:
        prev = self._history[-1] if self._history else None
        self._history.append(s)
        self._history = [h for h in self._history
                         if h.t >= s.t - max(self.damage_window_s, 5.0)]
        if s.t - self._last_fire_t < self.cooldown_s:
            return None
        # Stock loss
        if prev:
            for who in ("p1", "p2"):
                if s.stocks[who] < prev.stocks[who]:
                    self._last_fire_t = s.t
                    return TriggerEvent("stock_loss", who, s.t)
        # Damage spike over window
        window = [h for h in self._history if h.t >= s.t - self.damage_window_s]
        if len(window) >= 2:
            base = window[0]
            for who in ("p1", "p2"):
                if s.damage[who] - base.damage[who] >= self.damage_delta:
                    self._last_fire_t = s.t
                    return TriggerEvent("exchange", who, s.t)
        return None
```

- [ ] **Step 3: Run tests, commit**

```bash
pytest tests/test_trigger.py -v
git add mac/trigger.py tests/test_trigger.py
git commit -m "feat(mac): event trigger detector"
```

---

### Task 5.2: Rule-based scorer [B]

**Files:**
- Create: `mac/scorer.py`
- Test: `tests/test_scorer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_scorer.py
from mac.frame_data import FrameData
from mac.scorer import score_counterfactual

def test_score_punish_window():
    fd = FrameData.load("data/frame_data.json", "data/hitboxes.json")
    out = score_counterfactual(fd, attacker_char="toon_link",
                               attacker_move="fair", defender_char="joker",
                               defender_response="spotdodge")
    assert "punish_window_frames" in out
    assert isinstance(out["punish_window_frames"], int)
```

- [ ] **Step 2: Implement**

```python
# mac/scorer.py
from mac.frame_data import FrameData

# Joker's fastest OoS options, frame data (Smash Ultimate)
FASTEST_PUNISH = {
    "joker": {"OoS": 6,    # up-B
              "ground_dodge_to_attack": 4},
    "toon_link": {"OoS": 6, "ground_dodge_to_attack": 5},
}

def score_counterfactual(fd: FrameData,
                         attacker_char: str, attacker_move: str,
                         defender_char: str, defender_response: str) -> dict:
    """Return deterministic frame-counts the rewind card displays."""
    out = {"punish_window_frames": 0, "frame_advantage": None,
           "notes": []}
    try:
        m = fd.move(attacker_char, attacker_move)
    except KeyError:
        out["notes"].append(f"no frame data for {attacker_char}.{attacker_move}")
        return out
    if defender_response == "shield" and m.shield_advantage is not None:
        adv_for_defender = -m.shield_advantage
        out["frame_advantage"] = adv_for_defender
        if adv_for_defender > 0:
            out["punish_window_frames"] = adv_for_defender - \
                FASTEST_PUNISH[defender_char]["OoS"]
    elif defender_response in ("spotdodge", "roll", "airdodge"):
        if m.active_f and m.endlag_f:
            recovery_after_active = m.endlag_f - m.active_f[1]
            out["punish_window_frames"] = max(
                0, recovery_after_active -
                   FASTEST_PUNISH[defender_char]["ground_dodge_to_attack"])
    return out
```

- [ ] **Step 3: Run tests, commit**

```bash
pytest tests/test_scorer.py -v
git add mac/scorer.py tests/test_scorer.py
git commit -m "feat(mac): rule-based counterfactual scorer"
```

---

### Task 5.3: System 2 client + saliency keyframe selection [A]

**Files:**
- Create: `mac/system2_client.py`

- [ ] **Step 1: Implement**

```python
# mac/system2_client.py
import base64, cv2, httpx
from mac.state import StateT
from mac.trigger import TriggerEvent

def select_keyframes(buf: list[tuple[float, "any"]],
                     trajectory: list[StateT],
                     max_n: int = 8) -> list[tuple[float, "any"]]:
    """
    buf: list of (t, raw_frame_bgr) saved by main loop.
    Pick frames at action onsets in the trajectory + always the last frame.
    """
    onset_times = sorted({s.actions[w].onset_estimate_t
                          for s in trajectory for w in ("p1","p2")
                          if s.actions[w].label not in ("neutral", "walk")})
    target_times = onset_times[-max_n+1:] + [trajectory[-1].t]
    picks = []
    for tt in target_times:
        # nearest in buf
        if not buf: break
        nearest = min(buf, key=lambda x: abs(x[0] - tt))
        picks.append(nearest)
    return picks[-max_n:]

class System2Client:
    def __init__(self, url: str):
        self.url = url
        self._client = httpx.AsyncClient(timeout=20.0)

    async def request(self, event: TriggerEvent,
                      trajectory: list[StateT],
                      keyframes: list[tuple[float, "any"]]) -> dict | None:
        kfs = []
        for t, img in keyframes:
            small = cv2.resize(img, (640, 640))
            _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            kfs.append({"image_b64": base64.b64encode(jpg.tobytes()).decode(),
                        "t": t})
        try:
            r = await self._client.post(self.url, json={
                "state_trajectory": [s.model_dump() for s in trajectory],
                "keyframes": kfs,
                "event_type": event.kind,
            })
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError:
            return None
```

- [ ] **Step 2: Commit**

```bash
git add mac/system2_client.py
git commit -m "feat(mac): system2 client + saliency keyframe selection"
```

---

### Task 5.4: Rewind card UI [B]

**Files:**
- Create: `mac/rewind_card.py`

- [ ] **Step 1: Implement using OpenCV window**

```python
# mac/rewind_card.py
import cv2
import numpy as np
from mac.scorer import score_counterfactual
from mac.frame_data import FrameData

def render_card(keyframes: list[tuple[float, "any"]],
                response: dict,
                fd: FrameData,
                char_map: dict[str, str]) -> "np.ndarray":
    """
    response: System 2 JSON output.
    Returns 1280x720 image with rewind card.
    """
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Top strip: 4 keyframes
    n = min(4, len(keyframes))
    tile_w = 1280 // max(1, n)
    for i in range(n):
        _, img = keyframes[-(n-i)]
        thumb = cv2.resize(img, (tile_w, 280))
        canvas[20:300, i*tile_w:(i+1)*tile_w] = thumb
    # Summary
    cv2.putText(canvas, response.get("summary", "")[:80],
                (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    # Chosen vs counterfactual
    chosen = response["chosen_action"]
    cf = response["counterfactual_action"]
    cv2.putText(canvas, f"You did: {chosen['action_label']}",
                (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80,80,255), 2)
    cv2.putText(canvas, f"Alt: {cf['action_label']} ({cf['qualitative_likelihood']})",
                (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80,255,80), 2)
    # Deterministic scoring (if applicable)
    attacker_who = "p2" if chosen["player"] == "p1" else "p1"
    score = score_counterfactual(
        fd,
        attacker_char=char_map[attacker_who],
        attacker_move=chosen["action_label"],
        defender_char=char_map[chosen["player"]],
        defender_response=cf["action_label"],
    )
    pw = score.get("punish_window_frames", 0)
    if pw > 0:
        cv2.putText(canvas, f"Would have opened a {pw}-frame punish window",
                    (20, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,80), 2)
    # Citations
    y = 540
    for cit in response.get("frame_data_citations", [])[:4]:
        line = f"  {cit['character']} {cit['move']}: {cit['stat']}={cit['value']}"
        cv2.putText(canvas, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
        y += 30
    return canvas

def show_card(card: "np.ndarray", duration_s: float = 6.0):
    cv2.imshow("rewind-card", card)
    cv2.waitKey(int(duration_s * 1000))
    cv2.destroyWindow("rewind-card")
```

- [ ] **Step 2: Commit**

```bash
git add mac/rewind_card.py
git commit -m "feat(mac): rewind card UI with deterministic scoring"
```

---

### Task 5.5: Wire System 2 into main loop [joint]

**Files:**
- Modify: `mac/main.py`

- [ ] **Step 1: Add frame buffer + trigger + System 2 dispatch**

In `mac/main.py`, add at the top of `run()`:

```python
from collections import deque
from mac.trigger import TriggerDetector
from mac.system2_client import System2Client, select_keyframes
from mac.rewind_card import render_card, show_card

trigger = TriggerDetector(damage_delta=30.0, damage_window_s=2.0, cooldown_s=5.0)
s2 = System2Client(url=os.getenv("S2_URL", "http://localhost:8002/counterfactual"))
raw_frames: deque[tuple[float, "any"]] = deque(maxlen=600)  # ~10s at 60Hz
pending_s2: asyncio.Task | None = None
```

Inside the frame loop, after `buf.push(s)`:

```python
raw_frames.append((t, frame.img.copy()))
ev = trigger.check(s)
if ev is not None and (pending_s2 is None or pending_s2.done()):
    window_states = buf.window(ev.t - 5.0, ev.t)
    keyframes = select_keyframes(list(raw_frames), window_states, max_n=8)
    pending_s2 = asyncio.create_task(s2.request(ev, window_states, keyframes))

if pending_s2 is not None and pending_s2.done():
    resp = pending_s2.result()
    pending_s2 = None
    if resp is not None:
        card = render_card(keyframes, resp, fd, CHAR_MAP)
        show_card(card, duration_s=6.0)
```

- [ ] **Step 2: Manual end-to-end test**

```bash
S1_URL=http://<node>:8001/infer S2_URL=http://<node>:8002/counterfactual python -m mac.main
```

Play a stock. Confirm:
1. HUD updates live (damage, action labels, hitboxes).
2. On stock loss / big damage exchange, a rewind card pops up within ~5–10s, populated with citations.

- [ ] **Step 3: Commit (cut line H18)**

```bash
git add mac/main.py
git commit -m "feat(mac): wire System 2 trigger + rewind card into main loop"
```

---

## Phase 6: Polish + dry-run (H18–H21)

### Task 6.1: Latency audit [joint]

- [ ] **Step 1: Add timing breadcrumbs to main loop**

Add `time.monotonic()` around: capture frame, Tier 0, Tier 1, S1 dispatch return, HUD draw. Log every 30 frames. Identify any stage >50ms.

- [ ] **Step 2: Tune**

Common culprits and fixes:
- Tier 1 HSV detection slow on full-res → run on a 960×540 downscale.
- OCR slow → cache result for 100ms (damage doesn't change every frame).
- `cv2.imshow` is the slowest step on some Macs → expect 16–25ms; acceptable.

- [ ] **Step 3: Commit fixes if any**

```bash
git commit -am "perf: latency tuning"
```

---

### Task 6.2: Prompt iteration [B]

- [ ] **Step 1: Replay a sample match clip through System 1**

Save 20 sample frames covering common actions. Send each through `/infer`. Look at failure modes (e.g., model labels a dash as "walk", or never returns "fsmash").

- [ ] **Step 2: Tighten the System 1 prompt**

Add explicit visual cues per character if helpful:
```
Joker's f-smash has a dramatic forward sword swing with red flash.
Toon Link's bomb-pull is a static pose with a bomb visible in hand.
```

- [ ] **Step 3: Iterate prompt → smoke-test → repeat (timebox 60 min)**

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(prompts): tighten S1 vision cues for Joker + Toon Link"
```

---

### Task 6.3: Failure-mode dry runs [joint]

- [ ] **Step 1: Pull the network during a match**

Confirm Tier 0/1 HUD keeps rendering. Confirm no crashes.

- [ ] **Step 2: Force System 2 to return malformed JSON**

Temporarily break the schema in `server/prompts/system2.py` and confirm the client falls through gracefully (no rewind card, but HUD continues).

- [ ] **Step 3: Stock-loss while System 2 is mid-call**

Confirm cooldown logic prevents double-firing.

- [ ] **Step 4: Commit any fixes**

```bash
git commit -am "fix: hardening for demo failure modes"
```

---

## Phase 6.5: Optimization passes (H12–H16, parallel with prompt iteration)

After the baseline pipeline is working end-to-end, do a latency audit (Task 6.1) and pick the top 2–3 wins below based on what's actually hurting. Do NOT do all five — the spec's YAGNI directive holds even here.

### Task 6.5.1: Verify prefix caching is active

**Files:** none — verification only.

The System 1 prompt (~1.5k tokens) and System 2 prompt + frame-data blob (~6k tokens) are *identical* across every call. Prefix caching makes those tokens near-free after the first call.

- [ ] **Step 1: Confirm vLLM prefix caching**

vLLM v0.6+ enables prefix caching by default. To verify:

```bash
# On the node, in the vLLM server logs at startup, look for:
#   "Automatic prefix caching is enabled"
# Or pass --enable-prefix-caching explicitly in launch_system1.sh just to be safe.
```

Run two back-to-back `/infer` calls and time them. The second call's TTFT should drop by ~3–10×.

- [ ] **Step 2: Confirm SGLang RadixAttention**

SGLang's RadixAttention is always on. To verify: post two identical `/counterfactual` requests, time both. Second should be materially faster.

### Task 6.5.2: Multi-frame stack input to System 1

**Files:**
- Modify: `mac/dispatcher.py`
- Modify: `server/system1_server.py`
- Modify: `server/prompts/system1.py`

Single-frame phase detection (startup / active / endlag) is the weakest link in the spec. Qwen2.5-VL natively accepts multi-image input. Stacking 3 frames at ~100ms apart gives the model temporal context that dramatically sharpens phase calls and consequently hitbox onset accuracy.

- [ ] **Step 1: Buffer frames in dispatcher**

In `mac/dispatcher.py`, change `System1Client.maybe_infer` to buffer the most recent 3 frames at ~100ms intervals; when the rate-limit window allows, POST all 3 as a list. Adjust the API contract to `{"images_b64": [...], "ts": [...]}`.

- [ ] **Step 2: Accept image list on server**

In `server/system1_server.py`, change `FrameIn` to `{"images_b64": list[str], "ts": list[float]}` and build a chat prompt with all images.

- [ ] **Step 3: Update system prompt**

In `server/prompts/system1.py`, add to `SYSTEM_PROMPT`:

```
You receive 3 frames sampled at ~100ms intervals (oldest first). Use the temporal context to disambiguate move phases: a smash attack with the same pose across 2 frames is mid-active; transitions between distinct poses are startup or endlag.
```

### Task 6.5.3: Crop UI strip before sending to S1

**Files:**
- Modify: `mac/dispatcher.py`

The bottom ~15% of every frame is the damage/stocks UI (already extracted by Tier 0). The top ~5% is the timer. Both are redundant pixels in the VLM. Crop to the play area before downsampling to 640×640.

- [ ] **Step 1: Crop in dispatcher**

In `System1Client.maybe_infer`, before `cv2.resize`:

```python
h, w = img.shape[:2]
play = img[int(h*0.05):int(h*0.85), :]
small = cv2.resize(play, (640, 640))
```

### Task 6.5.4: Action label smoothing

**Files:**
- Create: `mac/smoother.py`
- Modify: `mac/main.py`

Single-frame action labels flicker. A 3-frame majority vote stabilizes the HUD and the trajectory we feed System 2.

- [ ] **Step 1: Write smoother**

```python
# mac/smoother.py
from collections import Counter, deque

class LabelSmoother:
    def __init__(self, k: int = 3):
        self.k = k
        self._hist = {"p1": deque(maxlen=k), "p2": deque(maxlen=k)}

    def update(self, who: str, label: str) -> str:
        self._hist[who].append(label)
        return Counter(self._hist[who]).most_common(1)[0][0]
```

- [ ] **Step 2: Wire into main loop**

In `mac/main.py`, instantiate `LabelSmoother(k=3)` and call `smoother.update(who, label)` on the raw S1 output before passing to `ActionState`.

### Task 6.5.5: Adaptive sampling on neutral

**Files:**
- Modify: `mac/dispatcher.py`

If both players are in neutral two calls in a row, double the next call's interval. Reset on any non-neutral label. Reclaims ~20–40% throughput during lulls; the spare cycles go to System 2's queue.

- [ ] **Step 1: Track consecutive-neutral counter**

In `System1Client`, add `_neutral_streak: int = 0`. On each successful infer, if both p1.action_label and p2.action_label are "neutral", increment; else reset.

- [ ] **Step 2: Apply adaptive interval**

`effective_interval = self.min_interval * (2 ** min(self._neutral_streak, 3))`.

---

## Phase 7: Stretch (H21–H23, only if green)

### Task 7.1: TTS post-stock (stretch)

**Files:**
- Modify: `mac/rewind_card.py`

- [ ] **Step 1: Add OpenAI TTS call**

```python
# in rewind_card.py
from openai import OpenAI
import os, tempfile, subprocess

def speak(text: str):
    if not os.getenv("OPENAI_API_KEY"): return
    client = OpenAI()
    speech = client.audio.speech.create(
        model="tts-1", voice="onyx", input=text[:200])
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(speech.content); path = f.name
    subprocess.Popen(["afplay", path])
```

In `show_card`, call `speak(response["summary"])` before `cv2.waitKey`.

- [ ] **Step 2: Verify, commit**

```bash
git commit -am "feat(stretch): TTS post-stock voice line"
```

### Task 7.2: Speculative decoding draft model on System 2 (stretch)

Wire SGLang `--draft-model` flag pointing at Qwen2.5-VL-7B on GPUs 6,7. Test that S2 latency drops. If wiring takes >2h, abandon and revert.

---

## Phase 8: Rehearsal (H23–H24)

- [ ] **Step 1: 3 full demo run-throughs.**
- [ ] **Step 2: Final commit + tag.**

```bash
git tag demo-final
```

---

## Demo-day checklist

- [ ] Mac plugged into wired Ethernet
- [ ] HD60 X passthrough to monitor confirmed
- [ ] Both H100 servers warm (curl them once)
- [ ] `OPENAI_API_KEY` set (if TTS shipped)
- [ ] `mac.main` runs cleanly for 60s with no Python tracebacks
- [ ] Rewind card window pre-positioned for visibility
- [ ] Backup: pre-recorded video of a working run, in case live demo fails
