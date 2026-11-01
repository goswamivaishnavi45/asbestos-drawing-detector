"""Text normalization for mixed English / Japanese drawing annotations."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Surface prefixes used on drawings to say *where* a material is used.
LOCATIONS = {
    "床": "floor", "巾木": "base", "幅木": "base", "壁": "wall", "腰壁": "wall",
    "天井": "ceiling", "屋根": "roof", "外壁": "exterior wall", "柱": "column", "梁": "beam", "配管": "piping",
    "floor": "floor", "base": "base", "wall": "wall", "ceiling": "ceiling", "piping": "piping",
    "roof": "roof", "ext wall": "exterior wall", "column": "column", "beam": "beam",
}
_ENUM_RE = re.compile(r"^\s*[(（]?\d{1,2}[.)）]\s*")  # "1. ", "(2) " note numbering
_LOC_RE = re.compile(
    r"^\s*(" + "|".join(sorted(map(re.escape, LOCATIONS), key=len, reverse=True)) + r")\s*[:：]\s*",
    re.IGNORECASE,
)

# Thickness / pitch specs, e.g. "t=12.5", "厚9.5mm", "12mm", "@455".
_SPEC_RE = re.compile(
    r"(t\s*=?\s*\d+(?:\.\d+)?\s*(?:mm)?|厚\s*\d+(?:\.\d+)?\s*(?:mm)?|\d+(?:\.\d+)?\s*mm|@\s*\d+)",
    re.IGNORECASE,
)

NON_ASBESTOS_MARKERS = ("ノンアスベスト", "ノンアス", "無石綿", "非石綿", "nonasbestos", "asbestosfree")
ASBESTOS_MARKERS = ("石綿含有", "アスベスト含有", "asbestoscontaining", "containsasbestos")

# Look-alike kanji that Japanese OCR confuses in safety-critical words (observed on degraded scans).
# Fixed explicitly rather than fuzzily: a fuzzy marker check once read "asbestos-containing" as "non-asbestos".
OCR_CONFUSIONS = {"合有": "含有", "石錦": "石綿", "石棉": "石綿"}

# Marker phrases are removed before matching so "Non-asbestos VAT" is looked up as "VAT".
_MARKER_RE = re.compile(
    r"[(（]?\s*(ノンアスベスト|ノンアス|無石綿|非石綿|石綿[含合]有|アスベスト[含合]有|"
    r"non[\s-]?asbestos|asbestos[\s-]?free|asbestos[\s-]?containing|contains asbestos)\s*[)）]?",
    re.IGNORECASE,
)

_STRIP_RE = re.compile(r"[\s\-‐―−_・･,，、。.．:：;；/／()（）\[\]［］「」『』【】<>＜＞\"'`~*#+=|]")


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def hira_to_kata(text: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


def normalize_key(text: str) -> str:
    """Canonical form used for matching: NFKC, lowercase, katakana, no punctuation/spaces."""
    t = hira_to_kata(nfkc(text)).lower()
    t = t.replace("ｰ", "ー")
    for wrong, right in OCR_CONFUSIONS.items():
        t = t.replace(wrong, right)
    return _STRIP_RE.sub("", t)


@dataclass
class ParsedAnnotation:
    raw: str
    location: str | None
    spec: str | None
    body: str  # material phrase with location and spec removed
    key: str  # normalize_key(body)


def parse_annotation(text: str) -> ParsedAnnotation:
    t = _ENUM_RE.sub("", nfkc(text).strip())
    location = None
    m = _LOC_RE.match(t)
    if m:
        location = LOCATIONS.get(m.group(1).lower(), LOCATIONS.get(m.group(1)))
        t = t[m.end():]
    if t.strip().lower() in LOCATIONS:  # a bare column header such as "天井" / "Floor"
        return ParsedAnnotation(text, LOCATIONS[t.strip().lower()], None, "", "")
    specs = [s.strip() for s in _SPEC_RE.findall(t)]
    body = _MARKER_RE.sub(" ", _SPEC_RE.sub(" ", t)).strip(" ,、")
    return ParsedAnnotation(text, location, " ".join(specs) or None, body, normalize_key(body))


def asbestos_marker(text: str) -> str | None:
    """'non' for explicit non-asbestos marking, 'asbestos' for explicit asbestos marking."""
    key = normalize_key(text)
    if any(m in key for m in NON_ASBESTOS_MARKERS):
        return "non"
    if any(m in key for m in ASBESTOS_MARKERS):
        return "asbestos"
    return None


# --- construction year -------------------------------------------------------

_ERA_BASE = {"明治": 1867, "大正": 1911, "昭和": 1925, "平成": 1988, "令和": 2018,
             "m": 1867, "t": 1911, "s": 1925, "h": 1988, "r": 2018}
# 竣 is often OCR'd as the look-alikes 峻 / 瑛 / 俊 on degraded scans.
_YEAR_KEYWORDS = r"(?:[竣峻瑛俊]工|新築|建築年|建設年|完成|築|completed|completion|built|year built|construction year|constructed)"
_ERA_RE = re.compile(r"(明治|大正|昭和|平成|令和)\s*(\d{1,2}|元)\s*年")
_KW_ERA_RE = re.compile(_YEAR_KEYWORDS + r"[^0-9a-z]{0,8}([mtshr])\s*(\d{1,2})\b", re.IGNORECASE)
_KW_YEAR_RE = re.compile(_YEAR_KEYWORDS + r"[^0-9]{0,15}((?:18|19|20)\d{2})", re.IGNORECASE)


def parse_build_year(text: str) -> int | None:
    """Find a construction year in title-block text. Supports Japanese era years (昭和53年, S53)."""
    t = nfkc(text)
    m = _KW_YEAR_RE.search(t)
    if m:
        return int(m.group(1))
    m = _ERA_RE.search(t)
    if m:
        n = 1 if m.group(2) == "元" else int(m.group(2))
        return _ERA_BASE[m.group(1)] + n
    m = _KW_ERA_RE.search(t)
    if m:
        return _ERA_BASE[m.group(1).lower()] + int(m.group(2))
    return None
