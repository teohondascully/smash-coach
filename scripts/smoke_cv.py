"""Smoke test for mac.tier1_cv.detect.

Loops capture, draws HSV-derived character bounding boxes, displays. Press 'q'
to quit.

Run:
    python scripts/smoke_cv.py
"""
from __future__ import annotations

import sys

import cv2

from mac.capture import Capture
from mac.tier1_cv import detect


def main() -> int:
    try:
        cap = Capture(device_index=0)
    except RuntimeError as e:
        print(f"[smoke_cv] {e}", file=sys.stderr)
        print(
            "[smoke_cv] No capture device available — this script requires "
            "an Elgato HD60 X (or equivalent UVC capture card) plugged in. "
            "Exiting cleanly.",
            file=sys.stderr,
        )
        return 0

    try:
        for f in cap.frames():
            bboxes = detect(f.img)
            for name, b in bboxes.items():
                if b is None:
                    continue
                cv2.rectangle(
                    f.img,
                    (int(b.x), int(b.y)),
                    (int(b.x + b.w), int(b.y + b.h)),
                    (0, 255, 0), 2,
                )
                cv2.putText(
                    f.img, name, (int(b.x), int(b.y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
            cv2.imshow("smash-coach: tier1 cv", f.img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
