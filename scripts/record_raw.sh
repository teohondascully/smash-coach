#!/usr/bin/env bash
# Record raw Cam Link feed to MP4 using the same capture config as mac.main.
# Output frame size, fps, and pixel layout are identical to what the live
# pipeline sees -- so a video recorded here can be played back through the
# pipeline as a CAP_DEV=<path> stand-in with no calibration drift.
#
# Usage:
#   ./scripts/record_raw.sh                          # writes ~/Desktop/gameplay.mp4, 90s max
#   OUT=/tmp/x.mp4 DURATION_S=120 ./scripts/record_raw.sh
#
# Press 'q' in the preview window to stop early.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-$HOME/Desktop/gameplay.mp4}"
CAP_DEV="${CAP_DEV:-0}"
DURATION_S="${DURATION_S:-90}"

uv run python - <<PY
import cv2, os, time
DEV = int(os.getenv("CAP_DEV", "$CAP_DEV"))
OUT = os.getenv("OUT", "$OUT")
DUR = float(os.getenv("DURATION_S", "$DURATION_S"))

cap = cv2.VideoCapture(DEV, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    print(f"failed to open capture device {DEV}"); raise SystemExit(1)

# Pull one frame so the codec sees real dimensions
for _ in range(8): cap.read()
ok, sample = cap.read()
if not ok:
    print("first read failed"); raise SystemExit(2)
h, w = sample.shape[:2]
print(f"capture: {w}x{h}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT, fourcc, 30.0, (w, h))
if not writer.isOpened():
    print(f"failed to open writer at {OUT}"); raise SystemExit(3)

cv2.namedWindow("recording (q=stop)", cv2.WINDOW_GUI_NORMAL)
print(f"recording -> {OUT}  | up to {DUR:.0f}s  | press q to stop early")

start = time.monotonic()
n = 0
while True:
    elapsed = time.monotonic() - start
    if elapsed >= DUR:
        break
    ok, frame = cap.read()
    if not ok:
        continue
    writer.write(frame)
    n += 1
    # Preview overlay
    overlay = frame.copy()
    txt = f"REC {elapsed:5.1f}/{DUR:.0f}s   frames={n}"
    cv2.putText(overlay, txt, (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0, 0, 255), 3)
    small = cv2.resize(overlay, (960, 540), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("recording (q=stop)", small)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

writer.release()
cap.release()
cv2.destroyAllWindows()
sz = os.path.getsize(OUT) / (1024 * 1024)
print(f"saved {OUT}  ({sz:.1f} MiB, {n} frames)")
PY

echo
echo "next: AirDrop / iMessage the file to your friend:"
echo "  $OUT"
