"""Deterministic template-matching digit OCR for SSBU damage % readouts.

The damage region is a 240x100 crop from a 1920x1080 frame. We binarize on
the bright glyph fill (HSV V channel -- works for white, yellow, orange,
and red damage colors), pull connected components belonging to the main
damage row, and template-match each one against canonical digit glyphs at
a normalized height. Decimal digits and the trailing percent sign are
ignored on purpose -- demo only needs integer-precision damage for
triggers and HUD readouts.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

CANONICAL_H = 48
MATCH_THRESHOLD = 0.5

# Component filters tuned for the 240x100 damage crop. The main digit row
# always sits at y in [37, 50] with h ~58-64; the small ".X" digit and the
# percent-sign cluster live below it. Width 8..55 excludes the wide "%"
# block (w ~60-75) and the thin antialiasing slivers.
MIN_DIGIT_H = 40
MAX_DIGIT_W = 55
MIN_DIGIT_W = 8
MIN_DIGIT_AREA = 100
MIN_DIGIT_Y = 25
MAX_DIGIT_Y = 55

# V-channel threshold -- captures white -> orange -> red glyphs alike.
# Black background / outlines stay at 0.
V_THRESHOLD = 100


def _binarize(crop_bgr: np.ndarray) -> np.ndarray:
    """Foreground = bright glyph-fill pixels (white/yellow/orange/red)."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    _, bd = cv2.threshold(hsv[:, :, 2], V_THRESHOLD, 255, cv2.THRESH_BINARY)
    return bd


def _digit_components(bin_img: np.ndarray) -> List[Tuple[int, int, int, int, np.ndarray]]:
    """Return main-row digit components as (x, y, w, h, glyph), sorted L->R."""
    n, _lbl, stats, _c = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    out: List[Tuple[int, int, int, int, np.ndarray]] = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if h < MIN_DIGIT_H or w > MAX_DIGIT_W or w < MIN_DIGIT_W:
            continue
        if a < MIN_DIGIT_AREA:
            continue
        if y < MIN_DIGIT_Y or y > MAX_DIGIT_Y:
            continue
        out.append((int(x), int(y), int(w), int(h), bin_img[y : y + h, x : x + w]))
    out.sort(key=lambda c: c[0])
    return out


def _normalize(g: np.ndarray, target_h: int = CANONICAL_H) -> np.ndarray:
    h, w = g.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_h // 2), dtype=np.uint8)
    new_w = max(1, int(round(w * target_h / h)))
    return cv2.resize(g, (new_w, target_h), interpolation=cv2.INTER_AREA)


class DigitMatcher:
    """Load digit templates and match damage-region glyphs against them."""

    def __init__(self, templates_dir: str = "data/digit_templates"):
        self.templates_dir = Path(templates_dir)
        self._templates: List[Tuple[str, np.ndarray]] = []
        self._load()

    def _load(self) -> None:
        if not self.templates_dir.exists():
            return
        for p in sorted(self.templates_dir.iterdir()):
            if p.suffix.lower() != ".png" or not p.stem.isdigit():
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Templates are stored white-glyph-on-black (same polarity as
            # _binarize output). Re-binarize defensively.
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            img = _normalize(img, CANONICAL_H)
            self._templates.append((p.stem, img))

    def _match_one(self, g: np.ndarray) -> Tuple[str, float]:
        if not self._templates:
            return "?", 0.0
        norm = _normalize(g, CANONICAL_H).astype(np.float32)
        best_ch, best = "?", -1.0
        for ch, tpl in self._templates:
            tf = tpl.astype(np.float32)
            if norm.shape[1] < tf.shape[1]:
                a = cv2.copyMakeBorder(
                    norm, 0, 0, 0, tf.shape[1] - norm.shape[1],
                    cv2.BORDER_CONSTANT, value=0,
                )
                b = tf
            else:
                a = norm
                b = tf
            res = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
            s = float(res.max())
            if s > best:
                best, best_ch = s, ch
        return best_ch, best

    def read(self, damage_crop: np.ndarray) -> Tuple[Optional[float], List[str]]:
        """Parse a damage crop into an integer-valued float (e.g. 47.0).

        Returns ``(value | None, glyphs)``. ``None`` if no digits were found
        or any glyph fell below ``MATCH_THRESHOLD``.
        """
        if damage_crop is None or damage_crop.size == 0:
            return None, []
        bin_img = _binarize(damage_crop)
        comps = _digit_components(bin_img)
        chars: List[str] = []
        for (_x, _y, _w, _h, g) in comps:
            ch, sc = self._match_one(g)
            chars.append(ch if sc >= MATCH_THRESHOLD else "?")
        if not chars or "?" in chars:
            return None, chars
        try:
            return float(int("".join(chars))), chars
        except ValueError:
            return None, chars
