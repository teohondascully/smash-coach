"""One-shot extractor for digit "8" from sample_1781926877.jpg p2 ("78").

The other digit templates were produced by an older pipeline; we just need
to drop in an "8.png" that matches the new matcher's polarity (white glyph
on black, normalized to height=48). Won't overwrite an existing 8.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mac.digit_match import (  # noqa: E402
    CANONICAL_H,
    _binarize,
    _digit_components,
    _normalize,
)

REGIONS_PATH = ROOT / "data" / "ui_regions.json"
TEMPLATES_DIR = ROOT / "data" / "digit_templates"
SOURCE_FN = "sample_1781926877.jpg"
SOURCE_WHO = "p2"  # "78%" -> rightmost main-row glyph is the 8


def _crop(img, regions, key: str):
    h, w = img.shape[:2]
    base_w, base_h = regions["resolution"]
    sx, sy = w / base_w, h / base_h
    x, y, ww, hh = regions[key]
    x, y, ww, hh = int(x * sx), int(y * sy), int(ww * sx), int(hh * sy)
    return img[y : y + hh, x : x + ww]


def main() -> None:
    out_path = TEMPLATES_DIR / "8.png"
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    regions = json.loads(REGIONS_PATH.read_text())
    img = cv2.imread(str(ROOT / "tests" / "fixtures" / SOURCE_FN))
    if img is None:
        raise SystemExit(f"could not read {SOURCE_FN}")
    crop = _crop(img, regions, f"{SOURCE_WHO}_damage")
    bin_img = _binarize(crop)
    comps = _digit_components(bin_img)
    if len(comps) < 2:
        raise SystemExit(
            f"expected >=2 main-row digits in {SOURCE_FN} {SOURCE_WHO}; got {len(comps)}"
        )
    x, y, w, h, g = comps[-1]
    norm = _normalize(g, CANONICAL_H)
    cv2.imwrite(str(out_path), norm)
    print(f"[ok] wrote {out_path} from {SOURCE_FN} {SOURCE_WHO} bbox=({x},{y},{w},{h})")


if __name__ == "__main__":
    main()
