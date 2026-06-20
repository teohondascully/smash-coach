"""Precompute a label track for a prerecorded clip using the accurate model.

Samples the clip at a fixed rate, sends multi-frame stacks to a System-1
server (point it at the 72B), and writes a timestamped track JSON that the
playback pipeline (LABEL_TRACK=...) looks up per displayed frame. Offline and
one-time, so per-call latency doesn't matter — we use the best model.

Usage:
    S1_URL=http://localhost:8003/infer \
    uv run python scripts/precompute_labels.py \
        --input data/gameplay.mov --out data/labels_track.json \
        --start 11 --interval 0.5 --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import cv2
import httpx

from mac.dispatcher import _prep


def _sample(input_path: str, start: float, end: float, interval: float,
            prep_size: int) -> list[dict]:
    """Walk the clip once, emit {video_t, images_b64, ts} stacks at ~interval."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    end = end if end > 0 else (total / fps if total else 1e9)
    step_frames = max(1, int(round(0.1 * fps)))  # ~0.1s spacing for the stack

    samples: list[dict] = []
    prev_b64: str | None = None
    prev_update = -1.0
    next_t = start
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        video_t = idx / fps
        idx += 1
        if video_t < start:
            continue
        if video_t > end:
            break
        # keep a frame from ~0.1s ago for temporal context
        if video_t - prev_update >= 0.1 or prev_b64 is None:
            cur = _prep(img, size=prep_size)
            if prev_b64 is None:
                prev_b64 = cur
            stack_prev = prev_b64
            prev_b64 = cur
            prev_update = video_t
        else:
            cur = _prep(img, size=prep_size)
            stack_prev = prev_b64
        if video_t >= next_t:
            samples.append({
                "video_t": round(video_t, 3),
                "images_b64": [stack_prev, cur],
                "ts": [round(video_t - 0.1, 3), round(video_t, 3)],
            })
            next_t += interval
    cap.release()
    return samples


async def _run(args) -> None:
    url = os.getenv("S1_URL", "http://localhost:8003/infer")
    print(f"[precompute] sampling {args.input} every {args.interval}s "
          f"({args.start}s..{args.end or 'end'}) prep={args.prep}px")
    samples = _sample(args.input, args.start, args.end, args.interval, args.prep)
    print(f"[precompute] {len(samples)} samples -> POST {url} "
          f"(concurrency {args.concurrency})")

    sem = asyncio.Semaphore(args.concurrency)
    entries: list[dict] = []
    done = 0
    t0 = time.time()

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def work(s: dict) -> None:
            nonlocal done
            async with sem:
                try:
                    r = await client.post(
                        url, json={"images_b64": s["images_b64"], "ts": s["ts"]}
                    )
                    r.raise_for_status()
                    d = r.json()
                    entries.append({"video_t": s["video_t"], "p1": d["p1"], "p2": d["p2"]})
                except Exception as e:  # noqa: BLE001
                    print(f"[precompute] t={s['video_t']} failed: {e}")
                finally:
                    done += 1
                    if done % 10 == 0 or done == len(samples):
                        el = time.time() - t0
                        print(f"[precompute] {done}/{len(samples)}  "
                              f"{el:.0f}s elapsed  {done/max(el,1e-6):.2f}/s")

        await asyncio.gather(*[work(s) for s in samples])

    entries.sort(key=lambda e: e["video_t"])
    out = {
        "meta": {
            "input": args.input, "url": url, "interval": args.interval,
            "start": args.start, "end": args.end, "prep": args.prep,
            "count": len(entries), "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "entries": entries,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[precompute] wrote {len(entries)} entries -> {args.out} "
          f"in {time.time()-t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/gameplay.mov")
    ap.add_argument("--out", default="data/labels_track.json")
    ap.add_argument("--start", type=float, default=11.0)
    ap.add_argument("--end", type=float, default=0.0)  # 0 = to end
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--prep", type=int, default=640)
    ap.add_argument("--timeout", type=float, default=300.0)
    asyncio.run(_run(ap.parse_args()))


if __name__ == "__main__":
    main()
