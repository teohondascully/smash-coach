"""Pod-side operations agent.

Tiny FastAPI app that exposes GPU, disk, and service health metrics to the
Mac-side dashboard. Designed to run on the Prime Intellect H100 pod under
`uvicorn server.ops_agent:app --port 9000`.

Lazy imports: anything not guaranteed on macOS is imported on demand so the
module can be imported on a Mac for smoke testing.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any

from fastapi import FastAPI

app = FastAPI(title="smash-coach-ops", version="0.1")


def _nvidia_smi_query() -> list[dict[str, Any]]:
    """Shell out to nvidia-smi and parse a CSV row per GPU.

    Returns a no-nvidia-smi stub when the binary isn't installed so the
    dashboard can still render on a Mac dev box.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=False
        )
    except FileNotFoundError:
        return [
            {
                "index": -1,
                "name": "no-nvidia-smi",
                "util_pct": 0,
                "mem_used_mib": 0,
                "mem_total_mib": 0,
                "temp_c": 0,
            }
        ]
    except subprocess.TimeoutExpired:
        return []

    gpus: list[dict[str, Any]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util_pct": int(float(parts[2])),
                    "mem_used_mib": int(float(parts[3])),
                    "mem_total_mib": int(float(parts[4])),
                    "temp_c": int(float(parts[5])),
                }
            )
        except ValueError:
            continue
    return gpus


def _probe(url: str) -> dict[str, Any]:
    """Ping a local FastAPI service for openapi.json. Returns {up, latency_ms}."""
    import httpx  # lazy

    t0 = time.perf_counter()
    try:
        r = httpx.get(url, timeout=1.5)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {"up": r.status_code == 200, "latency_ms": round(latency_ms, 2)}
    except Exception:
        return {"up": False, "latency_ms": None}


def _disk_usage(path: str = "/workspace") -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        used_path = path
    except (FileNotFoundError, OSError):
        usage = shutil.disk_usage("/")
        used_path = "/"
    gib = 1024**3
    return {
        "path": used_path,
        "used_gib": round(usage.used / gib, 2),
        "total_gib": round(usage.total / gib, 2),
        "free_gib": round(usage.free / gib, 2),
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"agent": "smash-coach-ops", "version": "0.1"}


@app.get("/gpu")
def gpu() -> list[dict[str, Any]]:
    return _nvidia_smi_query()


@app.get("/health/s1")
def health_s1() -> dict[str, Any]:
    return _probe("http://localhost:8001/openapi.json")


@app.get("/health/s2")
def health_s2() -> dict[str, Any]:
    return _probe("http://localhost:8002/openapi.json")


@app.get("/disk")
def disk() -> dict[str, Any]:
    return _disk_usage("/workspace")
