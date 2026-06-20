"""Smoke test for mac.tier0_ocr.Tier0.

Loops capture and prints damage / stocks readings per frame.

Run:
    python scripts/smoke_ocr.py
"""
from __future__ import annotations

import sys

from mac.capture import Capture
from mac.tier0_ocr import Tier0


def main() -> int:
    try:
        cap = Capture(device_index=0)
    except RuntimeError as e:
        print(f"[smoke_ocr] {e}", file=sys.stderr)
        print(
            "[smoke_ocr] No capture device available — this script requires "
            "an Elgato HD60 X (or equivalent UVC capture card) plugged in. "
            "Exiting cleanly.",
            file=sys.stderr,
        )
        return 0

    ocr = Tier0()
    try:
        for frame in cap.frames():
            d1 = ocr.damage(frame.img, "p1")
            d2 = ocr.damage(frame.img, "p2")
            s1 = ocr.stocks(frame.img, "p1")
            s2 = ocr.stocks(frame.img, "p2")
            print(
                f"p1: {d1}% ({s1} stocks)   p2: {d2}% ({s2} stocks)",
                end="\r",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
