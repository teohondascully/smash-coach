"""Smoke test for mac.capture.Capture.

Opens the default capture device and shows live frames in a cv2 window. Press
'q' to quit. If no capture card is attached, exits 0 with a friendly message.

Run:
    python scripts/smoke_capture.py
"""
from __future__ import annotations

import sys

import cv2

from mac.capture import Capture


def main() -> int:
    try:
        cap = Capture(device_index=0)
    except RuntimeError as e:
        print(f"[smoke_capture] {e}", file=sys.stderr)
        print(
            "[smoke_capture] No capture device available — this script "
            "requires an Elgato HD60 X (or equivalent UVC capture card) "
            "plugged in and granted camera permission. Exiting cleanly.",
            file=sys.stderr,
        )
        return 0

    try:
        for frame in cap.frames():
            cv2.imshow("smash-coach: capture", frame.img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
