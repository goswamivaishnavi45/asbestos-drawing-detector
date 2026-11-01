"""Text-line primitives and box utilities."""
from __future__ import annotations

from dataclasses import dataclass, replace

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


@dataclass
class TextLine:
    text: str
    bbox: Box
    conf: float = 1.0
    source: str = "ocr"  # "ocr" | "vector"

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def is_vertical(self) -> bool:
        return self.height > 1.5 * self.width


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def contains_point(box: Box, pt: tuple[float, float]) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _join(a: str, b: str) -> str:
    if a and b and a[-1].isascii() and a[-1].isalnum() and b[0].isascii() and b[0].isalnum():
        return f"{a} {b}"
    return a + b


def merge_lines(lines: list[TextLine], cells: list[Box] | None = None, gap_ratio: float = 0.6) -> list[TextLine]:
    """Merge horizontally adjacent fragments of the same text line.

    Fragments are never merged across table-cell borders, which keeps finish-schedule
    columns (floor / wall / ceiling) separate even when their text is close together.
    """
    cells = cells or []

    def cell_of(line: TextLine) -> int:
        return next((i for i, c in enumerate(cells) if contains_point(c, line.center)), -1)

    out = sorted(lines, key=lambda l: (l.bbox[0], l.bbox[1]))
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(out):
            if a.is_vertical:
                continue
            for j, b in enumerate(out):
                if j == i or b.is_vertical or b.bbox[0] < a.bbox[0]:
                    continue
                h = min(a.height, b.height)
                v_overlap = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
                gap = b.bbox[0] - a.bbox[2]
                if v_overlap < 0.6 * h or not (-0.3 * h <= gap <= gap_ratio * max(a.height, b.height)):
                    continue
                if cell_of(a) != cell_of(b):
                    continue
                merged = TextLine(
                    _join(a.text, b.text),
                    (a.bbox[0], min(a.bbox[1], b.bbox[1]), max(a.bbox[2], b.bbox[2]), max(a.bbox[3], b.bbox[3])),
                    min(a.conf, b.conf), a.source,
                )
                out = [l for k, l in enumerate(out) if k not in (i, j)] + [merged]
                out.sort(key=lambda l: (l.bbox[0], l.bbox[1]))
                changed = True
                break
            if changed:
                break
    return out


def offset(line: TextLine, dx: float, dy: float) -> TextLine:
    x0, y0, x1, y1 = line.bbox
    return replace(line, bbox=(x0 + dx, y0 + dy, x1 + dx, y1 + dy))
