"""Scrape ultimateframedata.com for Joker and Toon Link.

Writes data/frame_data.json with move entries keyed by canonical names from
data/action_vocab.json. Moves whose names cannot be mapped to a canonical name
are skipped. If anything fails, the script falls back to a hand-rolled baseline.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent
URLS = {
    "joker": "https://ultimateframedata.com/joker",
    "toon_link": "https://ultimateframedata.com/toon_link",
}

# Map raw scraped lowercased move-name (with spaces collapsed) -> canonical
COMMON_NAME_MAP: dict[str, str] = {
    "jab 1": "jab",
    "jab1": "jab",
    "jab": "jab",
    "neutral attack 1": "jab",
    "forward tilt": "ftilt",
    "ftilt": "ftilt",
    "up tilt": "utilt",
    "utilt": "utilt",
    "down tilt": "dtilt",
    "dtilt": "dtilt",
    "dash attack": "dashattack",
    "forward smash": "fsmash",
    "f-smash": "fsmash",
    "fsmash": "fsmash",
    "up smash": "usmash",
    "u-smash": "usmash",
    "usmash": "usmash",
    "down smash": "dsmash",
    "d-smash": "dsmash",
    "dsmash": "dsmash",
    "neutral air": "nair",
    "neutral aerial": "nair",
    "nair": "nair",
    "forward air": "fair",
    "forward aerial": "fair",
    "fair": "fair",
    "back air": "bair",
    "back aerial": "bair",
    "bair": "bair",
    "up air": "uair",
    "up aerial": "uair",
    "uair": "uair",
    "down air": "dair",
    "down aerial": "dair",
    "dair": "dair",
    "grab": "grab",
    "standing grab": "grab",
    "forward throw": "fthrow",
    "back throw": "bthrow",
    "up throw": "uthrow",
    "down throw": "dthrow",
}

JOKER_SPECIAL_MAP: dict[str, str] = {
    "eiha": "eiha",
    "eigaon": "eigaon",
    "tetrakarn": "tetrakarn",
    "makarakarn": "makarakarn",
    "gun": "gun",
    "gun special": "gun_special",
    "arsene": "arsene_summon",
    "rebellion": "wings_of_rebellion",
    "wings of rebellion": "wings_of_rebellion",
    "neutral special": "gun",
    "side special": "eiha",
    "up special": "wings_of_rebellion",
    "down special": "tetrakarn",
}

TOONLINK_SPECIAL_MAP: dict[str, str] = {
    "boomerang": "boomerang",
    "bomb": "bomb_pull",
    "bomb pull": "bomb_pull",
    "bomb throw": "bomb_throw",
    "arrow": "arrow",
    "hero's bow": "arrow",
    "spin attack": "spin_attack",
    "hookshot": "hookshot",
    "zair": "hookshot",
    "neutral special": "arrow",
    "side special": "boomerang",
    "up special": "spin_attack",
    "down special": "bomb_pull",
}

AERIAL_CANONICAL = {"nair", "fair", "bair", "uair", "dair"}
GROUND_CANONICAL = {
    "jab",
    "ftilt",
    "utilt",
    "dtilt",
    "dashattack",
    "fsmash",
    "usmash",
    "dsmash",
}
THROW_CANONICAL = {"fthrow", "bthrow", "uthrow", "dthrow"}


def _canonicalize(name_raw: str, char: str) -> Optional[str]:
    s = name_raw.lower().strip()
    s = re.sub(r"\s+", " ", s)
    if s in COMMON_NAME_MAP:
        return COMMON_NAME_MAP[s]
    sp = JOKER_SPECIAL_MAP if char == "joker" else TOONLINK_SPECIAL_MAP
    for key, canon in sp.items():
        if key in s:
            return canon
    return None


def _int_or_none(txt: str) -> Optional[int]:
    if not txt:
        return None
    m = re.search(r"-?\d+", txt)
    return int(m.group()) if m else None


def _active_range(txt: str) -> Optional[list[int]]:
    if not txt:
        return None
    # accept formats: "16-19", "16—19", "16, 21-23"
    txt = txt.replace("—", "-").replace("–", "-")
    nums = re.findall(r"\d+", txt)
    if not nums:
        return None
    return [int(nums[0]), int(nums[-1])]


def _category_for(canon: str) -> str:
    if canon in AERIAL_CANONICAL:
        return "aerial"
    if canon in GROUND_CANONICAL:
        return "ground"
    if canon == "grab":
        return "grab"
    if canon in THROW_CANONICAL:
        return "throw"
    return "special"


def _range_estimate(canon: str) -> str:
    if canon in {"jab", "dtilt"}:
        return "short"
    if canon in {"fsmash", "usmash", "dsmash", "ftilt", "dashattack"}:
        return "medium"
    if canon in {"arrow", "boomerang", "gun", "eiha", "eigaon", "bomb_throw", "hookshot"}:
        return "long"
    return "medium"


def parse_block(block, char: str) -> Optional[dict]:
    name_el = block.find(class_="movename")
    if not name_el:
        return None
    name_raw = name_el.get_text(" ", strip=True)
    canon = _canonicalize(name_raw, char)
    if not canon:
        return None

    def field(cls: str) -> str:
        el = block.find(class_=cls)
        return el.get_text(" ", strip=True) if el else ""

    startup = _int_or_none(field("startup"))
    active = _active_range(field("activeframes"))
    endlag = _int_or_none(field("endlag")) or _int_or_none(field("totalframes"))
    shield_adv = _int_or_none(field("advantage"))
    landing_lag = _int_or_none(field("landinglag"))
    if startup is None or active is None:
        return None
    return {
        "name": canon,
        "startup_f": startup,
        "active_f": active,
        "endlag_f": endlag if endlag is not None else (startup + (active[1] - active[0]) + 20),
        "landing_lag_f": landing_lag,
        "shield_advantage": shield_adv if shield_adv is not None else -8,
        "on_hit_advantage": None,
        "range_estimate": _range_estimate(canon),
        "category": _category_for(canon),
    }


def scrape_char(char: str, url: str) -> dict[str, dict]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    containers = soup.find_all(class_="movecontainer")
    out: dict[str, dict] = {}
    for block in containers:
        rec = parse_block(block, char)
        if rec is None:
            continue
        # first occurrence wins (e.g. "jab 1" over "jab 2")
        if rec["name"] not in out:
            out[rec["name"]] = rec
    return out


# ---- Hand-rolled fallback baseline (publicly known approximate values) ----

JOKER_FALLBACK: list[dict] = [
    {"name": "jab", "startup_f": 4, "active_f": [4, 5], "endlag_f": 18, "landing_lag_f": None, "shield_advantage": -16, "on_hit_advantage": None, "range_estimate": "short", "category": "ground"},
    {"name": "ftilt", "startup_f": 9, "active_f": [9, 11], "endlag_f": 30, "landing_lag_f": None, "shield_advantage": -7, "on_hit_advantage": None, "range_estimate": "medium", "category": "ground"},
    {"name": "utilt", "startup_f": 7, "active_f": [7, 11], "endlag_f": 27, "landing_lag_f": None, "shield_advantage": -10, "on_hit_advantage": None, "range_estimate": "short", "category": "ground"},
    {"name": "dtilt", "startup_f": 5, "active_f": [5, 6], "endlag_f": 18, "landing_lag_f": None, "shield_advantage": -7, "on_hit_advantage": None, "range_estimate": "short", "category": "ground"},
    {"name": "dashattack", "startup_f": 9, "active_f": [9, 14], "endlag_f": 39, "landing_lag_f": None, "shield_advantage": -12, "on_hit_advantage": None, "range_estimate": "medium", "category": "ground"},
    {"name": "fsmash", "startup_f": 16, "active_f": [16, 19], "endlag_f": 49, "landing_lag_f": None, "shield_advantage": -19, "on_hit_advantage": None, "range_estimate": "medium", "category": "ground"},
    {"name": "usmash", "startup_f": 14, "active_f": [14, 17], "endlag_f": 44, "landing_lag_f": None, "shield_advantage": -16, "on_hit_advantage": None, "range_estimate": "medium", "category": "ground"},
    {"name": "dsmash", "startup_f": 8, "active_f": [8, 10], "endlag_f": 43, "landing_lag_f": None, "shield_advantage": -17, "on_hit_advantage": None, "range_estimate": "medium", "category": "ground"},
    {"name": "nair", "startup_f": 5, "active_f": [5, 23], "endlag_f": 36, "landing_lag_f": 6, "shield_advantage": -7, "on_hit_advantage": None, "range_estimate": "medium", "category": "aerial"},
    {"name": "fair", "startup_f": 8, "active_f": [8, 11], "endlag_f": 35, "landing_lag_f": 12, "shield_advantage": -10, "on_hit_advantage": None, "range_estimate": "medium", "category": "aerial"},
    {"name": "bair", "startup_f": 7, "active_f": [7, 10], "endlag_f": 38, "landing_lag_f": 9, "shield_advantage": -8, "on_hit_advantage": None, "range_estimate": "medium", "category": "aerial"},
    {"name": "uair", "startup_f": 5, "active_f": [5, 9], "endlag_f": 34, "landing_lag_f": 9, "shield_advantage": -9, "on_hit_advantage": None, "range_estimate": "medium", "category": "aerial"},
    {"name": "dair", "startup_f": 9, "active_f": [9, 23], "endlag_f": 48, "landing_lag_f": 14, "shield_advantage": -12, "on_hit_advantage": None, "range_estimate": "medium", "category": "aerial"},
    {"name": "grab", "startup_f": 6, "active_f": [6, 7], "endlag_f": 33, "landing_lag_f": None, "shield_advantage": None, "on_hit_advantage": None, "range_estimate": "short", "category": "grab"},
    {"name": "eiha", "startup_f": 18, "active_f": [18, 22], "endlag_f": 60, "landing_lag_f": None, "shield_advantage": -20, "on_hit_advantage": None, "range_estimate": "long", "category": "special"},
    {"name": "eigaon", "startup_f": 22, "active_f": [22, 30], "endlag_f": 80, "landing_lag_f": None, "shield_advantage": -25, "on_hit_advantage": None, "range_estimate": "long", "category": "special"},
    {"name": "tetrakarn", "startup_f": 5, "active_f": [5, 30], "endlag_f": 50, "landing_lag_f": None, "shield_advantage": None, "on_hit_advantage": None, "range_estimate": "short", "category": "special"},
    {"name": "gun", "startup_f": 6, "active_f": [6, 8], "endlag_f": 32, "landing_lag_f": None, "shield_advantage": -10, "on_hit_advantage": None, "range_estimate": "long", "category": "special"},
    {"name": "arsene_summon", "startup_f": 30, "active_f": [99, 99], "endlag_f": 70, "landing_lag_f": None, "shield_advantage": None, "on_hit_advantage": None, "range_estimate": "short", "category": "special"},
    {"name": "wings_of_rebellion", "startup_f": 8, "active_f": [8, 12], "endlag_f": 50, "landing_lag_f": None, "shield_advantage": -20, "on_hit_advantage": None, "range_estimate": "medium", "category": "special"},
]

TOONLINK_FALLBACK: list[dict] = [
    {"name": "jab", "startup_f": 5, "active_f": [5, 6], "endlag_f": 20, "landing_lag_f": None, "shield_advantage": -8, "on_hit_advantage": None, "range_estimate": "short", "category": "ground"},
    {"name": "ftilt", "startup_f": 10, "active_f": [10, 12], "endlag_f": 31, "landing_lag_f": None, "shield_advantage": -10, "on_hit_advantage": None, "range_estimate": "medium", "category": "ground"},
    {"name": "utilt", "startup_f": 6, "active_f": [6, 11], "endlag_f": 27, "landing_lag_f": None, "shield_advantage": -9, "on_hit_advantage": None, "range_estimate": "short", "category": "ground"},
    {"name": "dtilt", "startup_f": 9, "active_f": [9, 10], "endlag_f": 23, "landing_lag_f": None, "shield_advantage": -6, "on_hit_advantage": None, "range_estimate": "short", "category": "ground"},
    {"name": "dashattack", "startup_f": 12, "active_f": [12, 16], "endlag_f": 40, "landing_lag_f": None, "shield_advantage": -11, "on_hit_advantage": None, "range_estimate": "medium", "category": "ground"},
    {"name": "fsmash", "startup_f": 15, "active_f": [15, 17], "endlag_f": 45, "landing_lag_f": None, "shield_advantage": -14, "on_hit_advantage": None, "range_estimate": "medium", "category": "ground"},
    {"name": "usmash", "startup_f": 13, "active_f": [13, 16], "endlag_f": 47, "landing_lag_f": None, "shield_advantage": -19, "on_hit_advantage": None, "range_estimate": "medium", "category": "ground"},
    {"name": "dsmash", "startup_f": 9, "active_f": [9, 11], "endlag_f": 49, "landing_lag_f": None, "shield_advantage": -18, "on_hit_advantage": None, "range_estimate": "medium", "category": "ground"},
    {"name": "nair", "startup_f": 7, "active_f": [7, 11], "endlag_f": 39, "landing_lag_f": 7, "shield_advantage": -7, "on_hit_advantage": None, "range_estimate": "medium", "category": "aerial"},
    {"name": "fair", "startup_f": 14, "active_f": [14, 16], "endlag_f": 43, "landing_lag_f": 11, "shield_advantage": -10, "on_hit_advantage": None, "range_estimate": "medium", "category": "aerial"},
    {"name": "bair", "startup_f": 6, "active_f": [6, 9], "endlag_f": 40, "landing_lag_f": 9, "shield_advantage": -8, "on_hit_advantage": None, "range_estimate": "medium", "category": "aerial"},
    {"name": "uair", "startup_f": 10, "active_f": [10, 13], "endlag_f": 41, "landing_lag_f": 10, "shield_advantage": -11, "on_hit_advantage": None, "range_estimate": "medium", "category": "aerial"},
    {"name": "dair", "startup_f": 16, "active_f": [16, 24], "endlag_f": 60, "landing_lag_f": 18, "shield_advantage": -20, "on_hit_advantage": None, "range_estimate": "medium", "category": "aerial"},
    {"name": "grab", "startup_f": 13, "active_f": [13, 27], "endlag_f": 70, "landing_lag_f": None, "shield_advantage": None, "on_hit_advantage": None, "range_estimate": "long", "category": "grab"},
    {"name": "boomerang", "startup_f": 14, "active_f": [14, 80], "endlag_f": 50, "landing_lag_f": None, "shield_advantage": -8, "on_hit_advantage": None, "range_estimate": "long", "category": "special"},
    {"name": "bomb_pull", "startup_f": 22, "active_f": [99, 99], "endlag_f": 40, "landing_lag_f": None, "shield_advantage": None, "on_hit_advantage": None, "range_estimate": "short", "category": "special"},
    {"name": "bomb_throw", "startup_f": 10, "active_f": [10, 60], "endlag_f": 38, "landing_lag_f": None, "shield_advantage": -6, "on_hit_advantage": None, "range_estimate": "long", "category": "special"},
    {"name": "arrow", "startup_f": 17, "active_f": [17, 50], "endlag_f": 55, "landing_lag_f": None, "shield_advantage": -10, "on_hit_advantage": None, "range_estimate": "long", "category": "special"},
    {"name": "spin_attack", "startup_f": 9, "active_f": [9, 25], "endlag_f": 70, "landing_lag_f": None, "shield_advantage": -25, "on_hit_advantage": None, "range_estimate": "medium", "category": "special"},
    {"name": "hookshot", "startup_f": 11, "active_f": [11, 25], "endlag_f": 60, "landing_lag_f": None, "shield_advantage": -15, "on_hit_advantage": None, "range_estimate": "long", "category": "special"},
]


def _merge_with_fallback(scraped: dict[str, dict], fallback: list[dict]) -> list[dict]:
    """Use scraped data when available, fall back for missing canonical names."""
    by_name = dict(scraped)
    for entry in fallback:
        if entry["name"] not in by_name:
            by_name[entry["name"]] = entry
    return list(by_name.values())


def main() -> None:
    out: dict[str, list[dict]] = {}
    fallbacks = {"joker": JOKER_FALLBACK, "toon_link": TOONLINK_FALLBACK}
    for char, url in URLS.items():
        scraped: dict[str, dict] = {}
        try:
            print(f"scraping {char}...", file=sys.stderr)
            scraped = scrape_char(char, url)
            print(f"  got {len(scraped)} canonical moves", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"  scrape failed: {e!r}; using fallback", file=sys.stderr)
        out[char] = _merge_with_fallback(scraped, fallbacks[char])
        time.sleep(1)
    (REPO_ROOT / "data/frame_data.json").write_text(json.dumps(out, indent=2))
    print(
        f"wrote {sum(len(v) for v in out.values())} moves "
        f"(joker={len(out['joker'])}, toon_link={len(out['toon_link'])})"
    )


if __name__ == "__main__":
    main()
