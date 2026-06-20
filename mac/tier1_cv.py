"""Tier 1 character bounding-box detection via HSV color masking.

Joker is keyed off his red coat, Toon Link off his green tunic. This is brittle
under heavy effects but workable on Final Destination. ``detect(img)`` returns
``{"p1": Bbox | None, "p2": Bbox | None}``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Bbox:
    x: float
    y: float
    w: float
    h: float
    cx: float
    cy: float


# tuned HSV ranges (adjust empirically against real capture)
JOKER_HSV = [(0, 80, 80), (10, 255, 255)]       # red coat
TOONLINK_HSV = [(40, 80, 80), (80, 255, 255)]   # green tunic

_MIN_AREA = 400


def _bbox_from_mask(mask) -> Optional[Bbox]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < _MIN_AREA:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return Bbox(
        float(x), float(y), float(w), float(h),
        float(x + w / 2), float(y + h / 2),
    )


def detect(img) -> dict:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out: dict = {}
    for name, (lo, hi) in [("p1", JOKER_HSV), ("p2", TOONLINK_HSV)]:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        out[name] = _bbox_from_mask(mask)
    return out
