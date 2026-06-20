#!/usr/bin/env bash
# Verify the HDMI capture card is detected and snapshot a frame.
# Usage: ./scripts/check_capture.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Enumerated cameras (macOS UVC) ==="
system_profiler SPCameraDataType 2>&1 | grep -E "Camera:|Model ID:" | head -20

echo
echo "=== USB device names containing 'cam', 'video', 'capture', 'elgato' ==="
ioreg -p IOUSB -l -w 0 2>&1 | grep -iE '"USB Product Name".*("cam|video|capture|elgato|hd60|game)' || echo "  (none found)"

echo
echo "=== Attempting frame grab on device 0, 1, 2 ==="
uv run python - <<'PY'
import cv2, os, sys
for idx in (0, 1, 2):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"  device {idx}: failed to open")
        continue
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    if not ok:
        print(f"  device {idx}: opened ({w}x{h}) but read failed")
        cap.release()
        continue
    out = f"/tmp/cap_dev{idx}.jpg"
    cv2.imwrite(out, frame)
    print(f"  device {idx}: opened ({w}x{h}), frame mean={frame.mean():.1f}, saved {out}")
    cap.release()
PY

echo
echo "=== Snapshot files ==="
ls -lh /tmp/cap_dev*.jpg 2>/dev/null || echo "  no snapshots"
echo
echo "Open the snapshots to verify which device is the Cam Link:"
echo "  open /tmp/cap_dev*.jpg"
