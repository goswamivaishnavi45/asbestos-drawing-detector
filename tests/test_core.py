import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asbestos_detector import ASBESTOS, NON_ASBESTOS, REVIEW, MaterialDB, MaterialMatcher  # noqa: E402
from asbestos_detector.geometry import TextLine, merge_lines  # noqa: E402
from asbestos_detector.metrics import cer, match_boxes, prf, wer  # noqa: E402
from asbestos_detector.normalize import asbestos_marker, normalize_key, parse_annotation, parse_build_year  # noqa: E402
from asbestos_detector.rules import classify  # noqa: E402


@pytest.fixture(scope="module")
def db():
    return MaterialDB.load()


@pytest.fixture(scope="module")
def matcher(db):
    return MaterialMatcher(db, "hybrid", threshold=0.75)


# --- normalization ---------------------------------------------------------------------

def test_normalize_key_unifies_width_and_kana():
    assert normalize_key("ＰＢ ボード") == normalize_key("pbぼーど") == "pbボード"


def test_parse_annotation_splits_location_and_spec():
    p = parse_annotation("天井：岩綿吸音板 t=12.5")
    assert p.location == "ceiling" and p.spec == "t=12.5" and p.body == "岩綿吸音板"


def test_parse_annotation_strips_numbering_and_markers():
    p = parse_annotation("2. 外壁: ECP(ノンアスベスト) 厚12mm")
    assert p.location == "exterior wall" and normalize_key(p.body) == "ecp"


def test_bare_header_is_not_a_material():
    assert parse_annotation("天井").key == "" and parse_annotation("Floor").key == ""


@pytest.mark.parametrize("text,year", [
    ("竣工: 昭和53年", 1978), ("竣工: 平成元年", 1989), ("竣工年: 1984年", 1984),
    ("Year built: 1972", 1972), ("完成 S48", 1973), ("日付 2023.04.10", None), ("竣工: 不明", None),
])
def test_parse_build_year(text, year):
    assert parse_build_year(text) == year


def test_markers():
    assert asbestos_marker("EP塗装(ノンアスベスト)") == "non"
    assert asbestos_marker("Non-asbestos VAT") == "non"
    assert asbestos_marker("石綿含有けい酸カルシウム板") == "asbestos"
    assert asbestos_marker("せっこうボード") is None


# --- matching ----------------------------------------------------------------------------

@pytest.mark.parametrize("text,entry_id", [
    ("壁：プラスターボード t=6", "A015"), ("天井：PB 9.5mm", "A015"), ("岩綿吸音板", "A014"),
    ("Ceiling: Rock wool acoustic board t=12", "A014"), ("Pタイル", "A016"), ("ケイカル板2種", "A005"),
    ("ビ二ル床タイル", "A016"),  # OCR confusion: 二 (kanji two) for ニ (katakana ni)
    ("ロックウ一ル吸音天井扳", "A014"),  # 一 for ー and 扳 for 板
    ("Non-asbestos VAT", "A016"),
])
def test_matcher_finds_material(matcher, text, entry_id):
    m = matcher.match(parse_annotation(text).body)
    assert m is not None and m.entry.id == entry_id


@pytest.mark.parametrize("text", ["事務室", "Corridor", "Toilet", "7800", "縮尺 S=1:200 図番 A-110", "仕上表", "Meeting Room"])
def test_matcher_rejects_non_materials(matcher, text):
    assert matcher.match(parse_annotation(text).body) is None


# --- rules -------------------------------------------------------------------------------

def test_rules(db):
    gypsum, sprayed, plywood = db.get("A015"), db.get("A001"), db.get("N002")
    assert classify(gypsum, "PB", 1978).label == ASBESTOS
    assert classify(gypsum, "PB", 1995).label == NON_ASBESTOS
    assert classify(gypsum, "PB", None).label == REVIEW
    assert classify(sprayed, "吹付け石綿", 2010).label == ASBESTOS
    assert classify(plywood, "合板", 1970).label == NON_ASBESTOS
    assert classify(gypsum, "PB(ノンアスベスト)", 1978).label == NON_ASBESTOS
    assert classify(None, "事務室", 1978) is None


# --- geometry + metrics -------------------------------------------------------------------

def test_merge_lines_respects_table_cells():
    a = TextLine("天井:", (0, 0, 30, 10))
    b = TextLine("岩綿吸音板", (33, 0, 90, 10))
    assert [l.text for l in merge_lines([a, b])] == ["天井:岩綿吸音板"]
    cells = [(0, 0, 31, 12), (31, 0, 100, 12)]
    assert len(merge_lines([a, b], cells)) == 2


def test_metrics():
    assert cer("せっこうボード", "せっこうボ一ド") == (1, 7)
    assert wer("rock wool board", "rock wol board") == (1, 3)
    assert match_boxes([(0, 0, 10, 10)], [(1, 1, 10, 10), (50, 50, 60, 60)]) == [(0, 0)]
    assert prf(8, 2, 0)["recall"] == 1.0
