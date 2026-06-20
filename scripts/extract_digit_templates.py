"""Extract canonical digit templates from the corpus.

Picks clean damage frames, segments main-row digits with the new
dark-outline pipeline (mac.digit_match._digit_components), and writes
``data/digit_templates/<digit>.png`` for each one observed. Existing
templates are NOT overwritten so a re-run won't disturb known-good
glyphs.

A ``_debug/`` directory is also populated with the raw crops + each
segmented component for human review.
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

# (fixture filename, who [p1/p2], expected main-row digit string).
# Only the main damage digits matter -- decimals are ignored.
SOURCES = [
    ("sample_1781924764.jpg", "p2", "12"),   # 1, 2
    ("sample_1781924776.jpg", "p2", "23"),   # 3
    ("sample_1781924782.jpg", "p2", "29"),   # 9
    ("sample_1781924803.jpg", "p2", "52"),   # 5
    ("sample_1781924800.jpg", "p2", "41"),   # 4
    ("sample_1781924770.jpg", "p1", "7"),    # 7
    ("sample_1781924776.jpg", "p1", "6"),    # 6
    ("sample_1781924758.jpg", "p2", "0"),    # 0
    ("sample_1781926877.jpg", "p2", "78"),   # 7, 8 -- 8 lives here
]

REGIONS_PATH = ROOT / "data" / "ui_regions.json"
TEMPLATES_DIR = ROOT / "data" / "digit_templates"
DEBUG_DIR = TEMPLATES_DIR / "_debug"


def _crop(img, regions, key: str):
    h, w = img.shape[:2]
    base_w, base_h = regions["resolution"]
    sx, sy = w / base_w, h / base_h
    x, y, ww, hh = regions[key]
    x, y, ww, hh = int(x * sx), int(y * sy), int(ww * sx), int(hh * sy)
    return img[y : y + hh, x : x + ww]


def main() -> None:
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    regions = json.loads(REGIONS_PATH.read_text())
    saved: dict[str, str] = {}

    for fn, who, expected in SOURCES:
        img_path = ROOT / "tests" / "fixtures" / fn
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] could not read {img_path}")
            continue
        crop = _crop(img, regions, f"{who}_damage")
        cv2.imwrite(str(DEBUG_DIR / f"{fn}_{who}_crop.png"), crop)
        bin_img = _binarize(crop)
        comps = _digit_components(bin_img)
        for idx, (x, y, w, h, g) in enumerate(comps):
            cv2.imwrite(str(DEBUG_DIR / f"{fn}_{who}_idx{idx:02d}_{w}x{h}.png"), g)
        if len(comps) != len(expected):
            print(
                f"[skip] {fn} {who}: expected {len(expected)} digits for "
                f"'{expected}', segmented {len(comps)}; review _debug/"
            )
            continue
        for ch, (x, y, w, h, g) in zip(expected, comps):
            out_path = TEMPLATES_DIR / f"{ch}.png"
            if out_path.exists():
                continue  # first capture wins
            cv2.imwrite(str(out_path), _normalize(g, CANONICAL_H))
            saved[ch] = f"{fn}/{who}"

    print("\nNewly extracted templates:")
    for ch, src in sorted(saved.items()):
        print(f"  {ch} <- {src}")
    have = sorted(p.stem for p in TEMPLATES_DIR.glob("*.png"))
    print(f"\nTemplates on disk ({len(have)}): {have}")


if __name__ == "__main__":
    main()
