#!/usr/bin/env bash
# Auto-capture a sample frame corpus for prompt iteration.
# Plays a live preview window AND saves a frame every N seconds.
# Default: 90 seconds total, 1 frame every 3s -> ~30 frames in tests/fixtures/.
# Press 'q' in the preview to stop early.
set -euo pipefail
cd "$(dirname "$0")/.."

CAP_DEV="${CAP_DEV:-0}"
DURATION_S="${DURATION_S:-90}"
INTERVAL_S="${INTERVAL_S:-3}"

mkdir -p tests/fixtures

uv run python - <<PY
import cv2, os, time

DEV         = int(os.getenv("CAP_DEV", "$CAP_DEV"))
DURATION_S  = float(os.getenv("DURATION_S", "$DURATION_S"))
INTERVAL_S  = float(os.getenv("INTERVAL_S", "$INTERVAL_S"))

cap = cv2.VideoCapture(DEV, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
if not cap.isOpened():
    print(f"failed to open device {DEV}")
    raise SystemExit(1)

print(f"corpus capture: {DURATION_S:.0f}s total, frame every {INTERVAL_S:.0f}s")
print(f"play diverse states (neutral, shielding, smashing, aerials, recovery)")
print(f"press 'q' in the preview window to stop early\n")

start = time.monotonic()
last_save = 0.0
saved = 0
while True:
    now = time.monotonic()
    elapsed = now - start
    if elapsed >= DURATION_S:
        break
    ok, frame = cap.read()
    if not ok:
        continue
    # auto-save at interval
    if now - last_save >= INTERVAL_S:
        ts = int(time.time())
        path = f"tests/fixtures/sample_{ts}.jpg"
        cv2.imwrite(path, frame)
        saved += 1
        last_save = now
        print(f"  [{elapsed:5.1f}s] saved {path}  (#{saved})")
    # preview
    overlay = frame.copy()
    txt = f"corpus {elapsed:.1f}/{DURATION_S:.0f}s   saved {saved}   next in {max(0, INTERVAL_S - (now-last_save)):.1f}s"
    cv2.putText(overlay, txt, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    small = cv2.resize(overlay, (1280, 720))
    cv2.imshow("corpus capture (q=quit)", small)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\ndone. corpus size: {saved} frames in tests/fixtures/")
PY
