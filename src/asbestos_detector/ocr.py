"""OCR backends. Each returns TextLine boxes in image pixel coordinates."""
from __future__ import annotations

import numpy as np

from .geometry import TextLine, iou, offset
from .preprocess import tiles


def _quad_to_box(pts) -> tuple[float, float, float, float]:
    xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _dedupe(lines: list[TextLine]) -> list[TextLine]:
    """Drop duplicates produced by overlapping tiles, keeping the higher-confidence read."""
    kept: list[TextLine] = []
    for line in sorted(lines, key=lambda l: -l.conf):
        if all(iou(line.bbox, k.bbox) < 0.5 for k in kept):
            kept.append(line)
    return kept


class OCRBackend:
    max_side = 2600  # larger sheets are tiled

    def _read(self, img: np.ndarray) -> list[TextLine]:
        raise NotImplementedError

    def read(self, img: np.ndarray) -> list[TextLine]:
        if max(img.shape[:2]) <= self.max_side:
            return self._read(img)
        out = []
        for x, y, w, h in tiles(img.shape, tile=1600, overlap=200):
            out += [offset(l, x, y) for l in self._read(img[y:y + h, x:x + w])]
        return _dedupe(out)


class EasyOCRBackend(OCRBackend):
    """CRAFT text detection + CRNN/CTC recognition, Japanese + English."""

    def __init__(self, langs=("ja", "en"), gpu: bool = False):
        import easyocr

        self.reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)

    def _read(self, img: np.ndarray) -> list[TextLine]:
        # rotation_info lets the recognizer try 90/270-degree crops, so vertical annotations are read.
        res = self.reader.readtext(img, detail=1, paragraph=False, rotation_info=[90, 270],
                                   canvas_size=2600, text_threshold=0.6, low_text=0.35)
        return [TextLine(text, _quad_to_box(pts), float(conf), "ocr") for pts, text, conf in res if text.strip()]


class PaddleOCRBackend(OCRBackend):
    """DBNet detection + SVTR recognition (PaddleOCR 2.x API, lang='japan'). Optional dependency."""

    def __init__(self, lang: str = "japan"):
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)

    def _read(self, img: np.ndarray) -> list[TextLine]:
        res = self.ocr.ocr(img, cls=True)[0] or []
        return [TextLine(text, _quad_to_box(pts), float(conf), "ocr") for pts, (text, conf) in res if text.strip()]


def get_backend(name: str) -> OCRBackend:
    if name == "easyocr":
        return EasyOCRBackend()
    if name == "paddleocr":
        return PaddleOCRBackend()
    raise ValueError(f"unknown OCR backend {name!r}")
