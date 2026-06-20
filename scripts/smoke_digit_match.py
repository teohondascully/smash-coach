"""Smoke harness for the template-matching digit OCR.

Runs ``Tier0.damage`` on each corpus frame, prints expected vs. actual,
and reports exact-match accuracy plus per-call latency. Damage is
compared as integers (47 == 47.0); decimals are intentionally ignored.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mac.tier0_ocr import Tier0  # noqa: E402
from mac.digit_match import DigitMatcher  # noqa: E402


GROUND_TRUTH: dict[str, dict[str, float | None]] = {
    "1781924758": {"p1": 0.0,  "p2": None},
    "1781924761": {"p1": 1.0,  "p2": None},
    "1781924764": {"p1": 10.0, "p2": 2.0},
    "1781924767": {"p1": 5.0,  "p2": 47.0},
    "1781924770": {"p1": 7.0,  "p2": 4.0},
    "1781924773": {"p1": 41.0, "p2": 23.0},
    "1781924776": {"p1": 6.0,  "p2": 7.2},
    "1781924785": {"p1": 16.0, "p2": 2.0},
    "1781924791": {"p1": 2.0,  "p2": 29.0},
    "1781924794": {"p1": 7.0,  "p2": 29.0},
    "1781924797": {"p1": 10.0, "p2": 35.0},
    "1781924800": {"p1": 4.0,  "p2": 41.0},
    "1781924803": {"p1": 0.0,  "p2": 44.0},
    "1781926877": {"p1": 77.0, "p2": 78.0},
    "1781926898": {"p2": 99.0},
    "1781926825": {"p1": 100.0},
    "1781926750": {"p1": 6.0,  "p2": 4.0},
    "1781926843": {"p1": 6.0,  "p2": 4.0},
    "1781926858": {"p1": 1.0},
}


def main() -> None:
    ocr = Tier0(regions_path=str(ROOT / "data" / "ui_regions.json"))
    fixtures = ROOT / "tests" / "fixtures"

    hits = 0
    misses = 0
    skipped = 0
    calls = 0
    total_ms = 0.0
    miss_rows: list[tuple[str, str, float, float | None]] = []

    for stem, expected in GROUND_TRUTH.items():
        path = fixtures / f"sample_{stem}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            print(f"[miss] could not read {path}")
            continue
        for who, want in expected.items():
            t0 = time.perf_counter()
            got = ocr.damage(img, who)
            total_ms += (time.perf_counter() - t0) * 1000
            calls += 1
            if want is None:
                skipped += 1
                continue
            ok = got is not None and int(got) == int(want)
            if ok:
                hits += 1
                tag = "OK"
            else:
                misses += 1
                miss_rows.append((stem, who, want, got))
                tag = "MISS"
            print(f"  {stem} {who}: got={got}  want={want}  {tag}")

    total = hits + misses
    pct = (100.0 * hits / total) if total else 0.0
    print(f"\nAccuracy: {hits}/{total} = {pct:.1f}%  (skipped {skipped})")
    print(f"Avg per Tier0.damage call: {total_ms / max(1, calls):.2f} ms  ({calls} calls)")

    # Pure DigitMatcher latency (no tesseract fallback) -- the hot-path number.
    matcher = DigitMatcher()
    sample = cv2.imread(str(fixtures / "sample_1781924767.jpg"))
    base_w, base_h = 1920, 1080
    h, w = sample.shape[:2]
    sx, sy = w / base_w, h / base_h
    x, y, ww, hh = 1280, 885, 240, 100
    cr = sample[int(y * sy) : int((y + hh) * sy), int(x * sx) : int((x + ww) * sx)]
    for _ in range(5):
        matcher.read(cr)
    t0 = time.perf_counter()
    N = 200
    for _ in range(N):
        matcher.read(cr)
    dt = (time.perf_counter() - t0) / N * 1000
    print(f"Pure DigitMatcher.read: {dt:.2f} ms / call")
    if miss_rows:
        print("\nMisses:")
        for stem, who, want, got in miss_rows:
            print(f"  {stem} {who}: want={want} got={got}")


if __name__ == "__main__":
    main()
