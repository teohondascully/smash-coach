"""Latency benchmark for System 1.

Sends N inferences against the same image and reports p50 / p95 / p99
plus mean. Critically, also checks whether prefix caching is working:
the second call onward should be materially faster than the first
(cold prompt + image encode vs. warm KV cache).

Usage:
    S1_URL=http://<pod>:8001/infer uv run python scripts/bench_s1.py
    S1_URL=http://localhost:8001/infer N=200 uv run python scripts/bench_s1.py
"""
from __future__ import annotations

import base64
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import httpx

S1_URL = os.getenv("S1_URL", "http://localhost:8001/infer")
N = int(os.getenv("N", "100"))
SAMPLE = os.getenv("SAMPLE", "tests/fixtures/sample_1781926877.jpg")
STACK_SIZE = int(os.getenv("STACK_SIZE", "3"))


def _b64(img) -> str:
    h = img.shape[0]
    top = int(h * 0.03)
    bot = int(h * 0.95)
    play = img[top:bot, :]
    small = cv2.resize(play, (640, 640))
    _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(jpg.tobytes()).decode()


def main() -> int:
    src = Path(SAMPLE)
    if not src.exists():
        print(f"missing sample frame: {src}", file=sys.stderr)
        return 1
    img = cv2.imread(str(src))
    if img is None:
        print(f"could not decode: {src}", file=sys.stderr)
        return 1

    b64 = _b64(img)
    payload = {
        "images_b64": [b64] * STACK_SIZE,
        "ts": [float(i) / 10 for i in range(STACK_SIZE)],
    }

    print(f"benchmark | url={S1_URL} | n={N} | stack={STACK_SIZE} | sample={SAMPLE}")
    print(f"warming up (1 call)...")
    with httpx.Client(timeout=30.0) as client:
        t0 = time.perf_counter()
        r = client.post(S1_URL, json=payload)
        cold_ms = (time.perf_counter() - t0) * 1000.0
        try:
            r.raise_for_status()
            cold_resp = r.json()
            print(f"  cold call: {cold_ms:.1f}ms  resp_keys={list(cold_resp.keys())}")
        except Exception as e:
            print(f"  cold call FAILED: {e}\n  {r.text[:300]}")
            return 2

        latencies_ms: list[float] = []
        for i in range(N):
            t0 = time.perf_counter()
            r = client.post(S1_URL, json=payload)
            dt = (time.perf_counter() - t0) * 1000.0
            try:
                r.raise_for_status()
            except Exception as e:
                print(f"  [{i}] FAIL {e}")
                continue
            latencies_ms.append(dt)
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{N}] last={dt:.1f}ms")

    if not latencies_ms:
        print("no successful calls")
        return 3

    latencies_ms.sort()
    mean = statistics.mean(latencies_ms)
    p50 = latencies_ms[len(latencies_ms) // 2]
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    p99 = latencies_ms[min(len(latencies_ms) - 1, int(len(latencies_ms) * 0.99))]
    speedup_vs_cold = cold_ms / mean if mean > 0 else 0.0

    print("\n=== results ===")
    print(f"  cold call : {cold_ms:7.1f} ms")
    print(f"  warm mean : {mean:7.1f} ms")
    print(f"  warm p50  : {p50:7.1f} ms")
    print(f"  warm p95  : {p95:7.1f} ms")
    print(f"  warm p99  : {p99:7.1f} ms")
    print(f"  cold/warm : {speedup_vs_cold:7.1f}x  (>2x => prefix caching is working)")

    if speedup_vs_cold < 2.0:
        print("\nWARN: cold/warm < 2x — verify enable_prefix_caching=True in S1 launch")
    if mean > 250.0:
        print("\nWARN: warm mean > 250ms — consider lower image res or fewer stack frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
