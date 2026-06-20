"""Streamlit ops dashboard for the Smash Coach pod.

Run:
    uv run streamlit run mac/ops_dashboard.py

Polls the pod-side ops agent (see server/ops_agent.py) on a configurable
interval and shows GPU, disk, cost, and service-health information.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import time
from datetime import datetime
from typing import Any

import streamlit as st

from mac.ops import CostTracker, PodLifecycle, PodMetrics

CONFIG_PATH = os.path.join("data", "ops_config.json")
STATE_PATH = os.path.join("data", "ops_state.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data
def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def fmt_runtime(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def fetch_snapshot(metrics: PodMetrics) -> dict[str, Any]:
    return asyncio.run(metrics.snapshot())


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="smash coach ops",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }

.stApp {
    background: #0a0a0a;
    color: #f5f5f5;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1400px !important;
}
html, body, [class*="css"], .stMarkdown, .stMarkdown p, .stMarkdown div {
    font-family: 'Inter', -apple-system, sans-serif;
    color: #f5f5f5;
}

/* hide widget labels we replace */
.stButton, .stProgress { margin: 0 !important; }

/* eyebrow / title row */
.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.35em;
    color: #555;
    font-weight: 500;
}
.title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 3rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.brand {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #f5f5f5;
}
.brand .accent { color: #00ff88; }

/* pill */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    text-transform: lowercase;
}
.pill .dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.pill.up   { color: #00ff88; background: rgba(0,255,136,0.06); border: 1px solid rgba(0,255,136,0.2); }
.pill.up .dot   { background: #00ff88; box-shadow: 0 0 8px rgba(0,255,136,0.6); }
.pill.down { color: #ff3b3b; background: rgba(255,59,59,0.06); border: 1px solid rgba(255,59,59,0.2); }
.pill.down .dot { background: #ff3b3b; }
.pill.idle { color: #777;    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); }
.pill.idle .dot { background: #555; }

/* big number stat */
.stat {
    padding: 1.5rem 0;
}
.stat .num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.6rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    line-height: 1;
    color: #f5f5f5;
}
.stat .num.accent { color: #00ff88; }
.stat .num.danger { color: #ff3b3b; }
.stat .num.muted  { color: #555; }
.stat .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.25em;
    color: #555;
    margin-top: 0.7rem;
}

/* section heading */
.section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.3em;
    color: #444;
    font-weight: 500;
    margin: 3rem 0 1.5rem 0;
}

/* progress bars (override streamlit) */
[data-testid="stProgress"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    height: 3px !important;
    border-radius: 2px !important;
}
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #00ff88 0%, #00d4ff 100%) !important;
}

/* buttons */
.stButton button {
    background: transparent !important;
    color: #999 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: lowercase !important;
    padding: 0.55rem 1.1rem !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
    font-weight: 400 !important;
    width: 100% !important;
}
.stButton button:hover {
    border-color: #00ff88 !important;
    color: #00ff88 !important;
    background: rgba(0,255,136,0.04) !important;
}
.stButton button:focus { box-shadow: none !important; outline: none !important; }

/* gpu card */
.gpucard {
    padding: 1.5rem 0;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.gpucard .name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #555;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.gpucard .grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
}
.gpucard .v {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    color: #f5f5f5;
    line-height: 1;
}
.gpucard .vsub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: #444;
    letter-spacing: 0.15em;
    margin-top: 0.5rem;
    text-transform: uppercase;
}
.gpucard .bar {
    height: 3px;
    background: rgba(255,255,255,0.04);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 0.6rem;
}
.gpucard .bar > div {
    height: 100%;
    background: linear-gradient(90deg, #00ff88 0%, #00d4ff 100%);
    transition: width 0.3s;
}

/* service row */
.service {
    padding: 1.5rem 0;
    border-top: 1px solid rgba(255,255,255,0.05);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.service .name {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    color: #f5f5f5;
}
.service .meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #666;
    letter-spacing: 0.05em;
}

/* footer */
.foot {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #444;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* cost strip below the row */
.coststrip {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #666;
    letter-spacing: 0.05em;
}
.coststrip .barwrap {
    flex: 1;
    height: 2px;
    background: rgba(255,255,255,0.04);
    border-radius: 1px;
    overflow: hidden;
}
.coststrip .barwrap > div {
    height: 100%;
    background: linear-gradient(90deg, #00ff88 0%, #00d4ff 100%);
    transition: width 0.3s;
}
.coststrip.danger .barwrap > div { background: #ff3b3b; }

/* alerts */
.stAlert {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 6px !important;
    color: #999 !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load config + clients
# ---------------------------------------------------------------------------

config = load_config()
agent_url: str = config["agent_url"]
rate_per_hour: float = float(config["rate_per_hour_usd"])
budget_usd: float = float(config["budget_usd"])
poll_interval_s: float = float(config["poll_interval_s"])

lifecycle = PodLifecycle()
metrics = PodMetrics(agent_url)
cost = CostTracker(STATE_PATH, rate_per_hour, budget_usd)

if "pod_id" not in st.session_state:
    st.session_state.pod_id = None


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

snap = fetch_snapshot(metrics)
runtime_s = cost.runtime_seconds()
spent = cost.spent_usd()
remaining = cost.remaining_usd()
burn = cost.burn_rate_per_hour()
pct = 0.0 if budget_usd <= 0 else min(1.0, spent / budget_usd)
spent_danger = spent > 0.75 * budget_usd
pod_id = st.session_state.pod_id

pod_pill = (
    f'<span class="pill up"><span class="dot"></span>pod {pod_id[:12] if pod_id else ""}</span>'
    if pod_id
    else '<span class="pill idle"><span class="dot"></span>no pod</span>'
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="title-row">
        <div>
            <div class="eyebrow">smash coach</div>
            <div class="brand">ops<span class="accent">.</span></div>
        </div>
        <div>{pod_pill}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Big stats row
# ---------------------------------------------------------------------------

cols = st.columns(4)
stats = [
    ("$" + f"{spent:.2f}", "spent", "danger" if spent_danger else "accent"),
    (fmt_runtime(runtime_s) if runtime_s > 0 else "—", "runtime", "" if runtime_s > 0 else "muted"),
    ("$" + f"{burn:.2f}/hr", "burn rate", "" if burn > 0 else "muted"),
    ("$" + f"{remaining:.2f}", "remaining", "muted" if remaining > 0.5 * budget_usd else "danger"),
]
for col, (num, label, cls) in zip(cols, stats):
    col.markdown(
        f"""
        <div class="stat">
          <div class="num {cls}">{num}</div>
          <div class="label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="coststrip {'danger' if spent_danger else ''}">
        <span>budget</span>
        <div class="barwrap"><div style="width: {pct*100:.1f}%"></div></div>
        <span>{pct*100:.1f}% of ${budget_usd:.0f}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

st.markdown('<div class="section">controls</div>', unsafe_allow_html=True)

bcols = st.columns(4)

if bcols[0].button("start pod"):
    pod_id = lifecycle.start()
    if pod_id:
        st.session_state.pod_id = pod_id
        cost.mark_started()
        st.success(f"started {pod_id}")
    else:
        st.error("start failed — check prime cli")

if bcols[1].button("stop pod"):
    pid = st.session_state.pod_id
    if not pid:
        st.error("no pod id")
    elif lifecycle.stop(pid):
        cost.mark_stopped()
        st.session_state.pod_id = None
        st.success(f"stopped {pid}")
    else:
        st.error("stop failed")

if bcols[2].button("restart s1"):
    pid = st.session_state.pod_id
    if not pid:
        st.error("no pod id")
    elif lifecycle.restart_service(pid, "smash-system1"):
        st.success("restart s1 sent")
    else:
        st.error("restart s1 failed")

if bcols[3].button("restart s2"):
    pid = st.session_state.pod_id
    if not pid:
        st.error("no pod id")
    elif lifecycle.restart_service(pid, "smash-system2"):
        st.success("restart s2 sent")
    else:
        st.error("restart s2 failed")


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

st.markdown('<div class="section">services</div>', unsafe_allow_html=True)


def render_service_row(name: str, h: dict | None) -> str:
    up = bool(h and h.get("up"))
    latency = h.get("latency_ms") if h else None
    lat_str = f"{latency:.0f}ms" if isinstance(latency, (int, float)) else "—"
    pill_cls = "up" if up else "down" if h else "idle"
    pill_text = "online" if up else "offline" if h else "idle"
    return f"""
    <div class="service">
        <div class="name">{name}</div>
        <div style="display:flex;align-items:center;gap:1.5rem">
            <div class="meta">{lat_str}</div>
            <span class="pill {pill_cls}"><span class="dot"></span>{pill_text}</span>
        </div>
    </div>
    """


services_html = render_service_row(
    "system 1 · perception", snap.get("s1")
) + render_service_row("system 2 · counterfactual", snap.get("s2"))
st.markdown(services_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# GPUs
# ---------------------------------------------------------------------------

st.markdown('<div class="section">gpus</div>', unsafe_allow_html=True)

gpus = snap.get("gpu") or []
if not gpus:
    st.info("no gpu data — pod offline or agent unreachable")
else:
    rows = [gpus[i : i + 4] for i in range(0, len(gpus), 4)]
    for row in rows:
        cols = st.columns(4)
        for col, g in zip(cols, row):
            idx = g.get("index", "?")
            name = g.get("name", "?")
            util = int(g.get("util_pct", 0))
            used = float(g.get("mem_used_mib", 0)) / 1024
            total = float(g.get("mem_total_mib", 0)) / 1024 or 1.0
            mem_pct = min(1.0, used / total)
            temp = int(g.get("temp_c", 0))
            short_name = name.replace("NVIDIA ", "").replace("GeForce ", "").strip()
            col.markdown(
                f"""
                <div class="gpucard">
                    <div class="name">gpu {idx} · {short_name}</div>
                    <div class="grid">
                        <div>
                            <div class="v">{util}%</div>
                            <div class="vsub">util</div>
                            <div class="bar"><div style="width:{util}%"></div></div>
                        </div>
                        <div>
                            <div class="v">{used:.1f}<span style="color:#444;font-size:0.7em"> /{total:.0f}</span></div>
                            <div class="vsub">vram gib</div>
                            <div class="bar"><div style="width:{mem_pct*100:.1f}%"></div></div>
                        </div>
                        <div>
                            <div class="v">{temp}°</div>
                            <div class="vsub">temp</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

disk = snap.get("disk")
if disk:
    used = float(disk.get("used_gib", 0.0))
    total = float(disk.get("total_gib", 0.0)) or 1.0
    free = float(disk.get("free_gib", total - used))
    path = disk.get("path", "/")
    disk_pct = min(1.0, used / total)
    st.markdown('<div class="section">disk</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="service">
            <div>
                <div class="name">{path}</div>
                <div class="meta">{used:.1f} / {total:.0f} gib used · {free:.0f} free</div>
            </div>
            <div style="flex:0 0 280px">
                <div class="bar" style="height:3px;background:rgba(255,255,255,0.04);border-radius:2px;overflow:hidden">
                    <div style="width:{disk_pct*100:.1f}%;height:100%;background:linear-gradient(90deg,#00ff88 0%,#00d4ff 100%)"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------

fixtures = sorted(glob.glob("tests/fixtures/sample_*.jpg"))
if fixtures:
    st.markdown(
        f'<div class="section">captures · {len(fixtures)}</div>',
        unsafe_allow_html=True,
    )
    cols_per_row = 4
    rows = [fixtures[i : i + cols_per_row] for i in range(0, len(fixtures), cols_per_row)]
    for row in rows:
        cols = st.columns(cols_per_row)
        for col, path in zip(cols, row):
            stem = os.path.basename(path).replace("sample_", "").replace(".jpg", "")
            size_kb = os.path.getsize(path) / 1024
            col.image(path, use_container_width=True)
            col.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#555;letter-spacing:0.1em;margin-top:-0.5rem;margin-bottom:1.5rem">{stem} · {size_kb:.0f}kb</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

ts = datetime.fromtimestamp(snap.get("fetched_at", time.time())).strftime("%H:%M:%S")
st.markdown(
    f"""
    <div class="foot">
        last refresh {ts} · poll {poll_interval_s:.0f}s · {agent_url}
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

time.sleep(poll_interval_s)
st.rerun()
