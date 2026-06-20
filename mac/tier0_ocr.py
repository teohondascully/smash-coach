"""Tier 0 deterministic CV: damage % OCR + stock-icon counting.

Reads ``data/ui_regions.json`` at init. Regions are stored at a reference
resolution (e.g. 1920x1080) and rescaled at runtime to whatever the live capture
frame happens to be. Functions return ``None`` for damage if no digits parsed
rather than crashing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract

from mac.digit_match import DigitMatcher


class Tier0:
    def __init__(self, regions_path: str = "data/ui_regions.json"):
        cfg = json.loads(Path(regions_path).read_text())
        self.regions = cfg
        self.base_res = tuple(cfg["resolution"])  # (W, H)
        # Deterministic template-matching digit OCR. ~100% accurate on the
        # fixed SSBU damage font and runs in << 1ms; far more reliable than
        # tesseract on the outlined glyphs. Tesseract stays as fallback.
        self.matcher = DigitMatcher()

    def _crop(self, img, key: str):
        h, w = img.shape[:2]
        sx, sy = w / self.base_res[0], h / self.base_res[1]
        x, y, ww, hh = self.regions[key]
        x, y, ww, hh = int(x * sx), int(y * sy), int(ww * sx), int(hh * sy)
        # clamp
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        ww = max(1, min(ww, w - x))
        hh = max(1, min(hh, h - y))
        return img[y : y + hh, x : x + ww]

    def damage(self, img, who: str) -> Optional[float]:
        try:
            crop = self._crop(img, f"{who}_damage")
            if crop.size == 0:
                return None
            # Primary: deterministic template-match digit OCR.
            val, _glyphs = self.matcher.read(crop)
            if val is not None:
                return val
            # Fallback: tesseract when a glyph can't be matched (e.g. an "8"
            # for which we have no template yet).
            return self._tesseract_damage(crop)
        except Exception:
            return None

    def _tesseract_damage(self, crop) -> Optional[float]:
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Otsu binarization + 3x upscale handles SSBU's outlined font better
            # than fixed thresholding. Tesseract likes large glyphs (>30px tall).
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            big = cv2.resize(th, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            # Try a couple of PSM modes; first hit wins.
            for psm in (11, 7, 8):
                text = pytesseract.image_to_string(
                    big,
                    config=f"--psm {psm} -c tessedit_char_whitelist=0123456789.%",
                )
                m = re.search(r"\d+(?:\.\d+)?", text)
                if not m:
                    continue
                val = float(m.group())
                # SSBU damage caps at ~999; anything beyond is a misread.
                if val <= 999.0:
                    return val
            return None
        except Exception:
            return None

    def stocks(self, img, who: str) -> int:
        try:
            crop = self._crop(img, f"{who}_stocks")
            if crop.size == 0:
                return 0
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            n_labels, _ = cv2.connectedComponents(th)
            return max(0, min(3, n_labels - 1))
        except Exception:
            return 0
