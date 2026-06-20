#!/usr/bin/env bash
# Live preview of the Cam Link feed.
# Leave this open while working — keeps the Switch awake.
# Press 'q' in the preview window to quit, or 's' to save the current frame.
set -euo pipefail
cd "$(dirname "$0")/.."

CAP_DEV="${CAP_DEV:-0}"
PREVIEW_W="${PREVIEW_W:-960}"
PREVIEW_H="${PREVIEW_H:-540}"

uv run python - <<PY
import cv2, os, time
DEV = int(os.getenv("CAP_DEV", "$CAP_DEV"))
W   = int(os.getenv("PREVIEW_W", "$PREVIEW_W"))
H   = int(os.getenv("PREVIEW_H", "$PREVIEW_H"))

cap = cv2.VideoCapture(DEV, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
# Force MJPEG to keep USB throughput modest; matters more on some hubs
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always read the most recent frame

if not cap.isOpened():
    print(f"failed to open device {DEV}"); raise SystemExit(1)

cv2.namedWindow("cam-link preview (q=quit, s=save)", cv2.WINDOW_GUI_NORMAL)

print(f"device {DEV} | preview {W}x{H} | press q to quit, s to save")
last = time.monotonic()
frames = 0
last_mean = 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        continue
    frames += 1
    now = time.monotonic()
    if now - last >= 2.0:
        # only compute brightness every 2s — saving ~10ms per frame
        last_mean = float(frame.mean())
        print(f"  fps={frames/(now-last):5.1f}  mean={last_mean:.1f}")
        frames = 0; last = now
    small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("cam-link preview (q=quit, s=save)", small)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    if k == ord("s"):
        ts = int(time.time())
        path = f"tests/fixtures/sample_{ts}.jpg"
        cv2.imwrite(path, frame)
        print(f"  saved {path}")

cap.release(); cv2.destroyAllWindows()
PY
