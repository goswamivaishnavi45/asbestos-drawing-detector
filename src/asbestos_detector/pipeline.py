"""End-to-end pipeline: drawing -> text lines -> materials -> asbestos determination."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .database import MaterialDB
from .geometry import Box, TextLine, contains_point, merge_lines
from .ingest import load_pages
from .layout import detect_tables
from .matcher import MaterialMatcher
from .normalize import asbestos_marker, parse_annotation, parse_build_year
from .ocr import OCRBackend, get_backend
from .preprocess import preprocess
from .rules import classify


@dataclass
class Item:
    page: int
    text: str
    bbox: Box
    source: str
    ocr_conf: float
    region: str  # "table" | "drawing"
    location: str | None = None
    spec: str | None = None
    material_id: str | None = None
    material: str | None = None
    match_score: float | None = None
    label: str | None = None  # None => text is not a material
    reason: str | None = None


@dataclass
class PageResult:
    index: int
    dpi: float
    mode: str  # "vector" | "ocr"
    skew_angle: float
    n_tables: int


@dataclass
class Report:
    file: str
    build_year: int | None
    build_year_source: str
    pages: list[PageResult] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)

    @property
    def materials(self) -> list[Item]:
        return [i for i in self.items if i.label]

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def save_csv(self, path: str | Path) -> None:
        cols = ["page", "text", "location", "spec", "material_id", "material", "match_score", "label", "reason", "region", "bbox"]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel shows Japanese correctly
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for it in self.materials:
                w.writerow({**asdict(it), "bbox": [round(v) for v in it.bbox]})


def find_build_year(lines: list[TextLine]) -> int | None:
    # Prefer a year found within one line; fall back to the joined page text.
    return next((y for y in (parse_build_year(l.text) for l in lines) if y), None) \
        or parse_build_year(" ".join(l.text for l in lines))


class Analyzer:
    def __init__(self, db: MaterialDB | None = None, ocr: str | OCRBackend = "easyocr",
                 match_method: str = "hybrid", threshold: float = 0.75, dpi: float = 150,
                 deskew: bool = True, denoise: bool = True):
        self.db = db or MaterialDB.load()
        self.matcher = MaterialMatcher(self.db, match_method, threshold)
        self._ocr = ocr
        self.dpi, self.deskew, self.denoise = dpi, deskew, denoise

    @property
    def ocr(self) -> OCRBackend:
        if isinstance(self._ocr, str):  # load model lazily: vector PDFs never need it
            self._ocr = get_backend(self._ocr)
        return self._ocr

    def read_page_lines(self, page, force_ocr: bool = False):
        """Returns (lines, processed image, skew angle, tables, mode)."""
        if page.has_vector_text and not force_ocr:
            img, angle, mode = page.image, 0.0, "vector"
            lines = page.vector_lines
        else:
            img, angle = preprocess(page.image, self.deskew, self.denoise)
            mode = "ocr"
            lines = self.ocr.read(img)
        tables = detect_tables(img)
        cells = [c for t in tables for c in t.cells]
        return merge_lines(lines, cells), img, angle, tables, mode

    def classify_lines(self, lines: list[TextLine], build_year: int | None, page_index: int = 0,
                       cells: list[Box] | None = None) -> list[Item]:
        items = []
        for line in lines:
            parsed = parse_annotation(line.text)
            in_table = any(contains_point(c, line.center) for c in (cells or []))
            item = Item(page_index, line.text, line.bbox, line.source, line.conf,
                        "table" if in_table else "drawing", parsed.location, parsed.spec)
            match = self.matcher.match(parsed.body) if len(parsed.key) >= 2 else None
            decision = classify(match.entry if match else None, line.text, build_year)
            if decision and (match or asbestos_marker(line.text)):
                item.label, item.reason = decision.label, decision.reason
                if match:
                    item.material_id, item.material = match.entry.id, match.entry.name_en
                    item.match_score = round(match.score, 3)
            items.append(item)
        return items

    def analyze(self, path: str | Path, build_year: int | None = None, force_ocr: bool = False) -> tuple[Report, list]:
        """Returns the report and the processed page images (for visualization)."""
        pages = load_pages(path, self.dpi)
        per_page = [self.read_page_lines(p, force_ocr) for p in pages]

        year_source = "user" if build_year else "not found"
        if build_year is None:
            build_year = next((y for y in (find_build_year(lines) for lines, *_ in per_page) if y), None)
            if build_year:
                year_source = "title block"

        report = Report(str(path), build_year, year_source)
        images = []
        for page, (lines, img, angle, tables, mode) in zip(pages, per_page):
            cells = [c for t in tables for c in t.cells]
            report.pages.append(PageResult(page.index, page.dpi, mode, round(angle, 2), len(tables)))
            report.items += self.classify_lines(lines, build_year, page.index, cells)
            images.append(img)
        return report, images
