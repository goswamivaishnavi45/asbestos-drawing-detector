"""Load drawings. Vector PDFs give exact text for free; scans and images go through OCR."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from .geometry import TextLine

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MIN_VECTOR_CHARS = 20  # fewer embedded characters than this => treat the page as a scan


@dataclass
class Page:
    index: int
    image: np.ndarray  # grayscale
    dpi: float
    vector_lines: list[TextLine] = field(default_factory=list)

    @property
    def has_vector_text(self) -> bool:
        return sum(len(l.text) for l in self.vector_lines) >= MIN_VECTOR_CHARS


def _vector_lines(page: pymupdf.Page, scale: float) -> list[TextLine]:
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                x0, y0, x1, y1 = line["bbox"]
                lines.append(TextLine(text, (x0 * scale, y0 * scale, x1 * scale, y1 * scale), 1.0, "vector"))
    return lines


def load_pages(path: str | Path, dpi: float = 150) -> list[Page]:
    path = Path(path)
    if path.suffix.lower() in IMAGE_EXTS:
        data = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII Windows paths
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"cannot read image {path}")
        return [Page(0, img, dpi)]
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"unsupported file type {path.suffix}")

    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=int(dpi), colorspace=pymupdf.csGRAY)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
            pages.append(Page(i, img, dpi, _vector_lines(page, dpi / 72)))
    return pages
