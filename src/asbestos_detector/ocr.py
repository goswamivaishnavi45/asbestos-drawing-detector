"""OCR backends. Each returns TextLine boxes in image pixel coordinates."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import TextLine, iou, offset
from .preprocess import tiles


def _quad_to_box(pts) -> tuple[float, float, float, float]:
    xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _quad_bounds(pts) -> tuple[int, int, int, int]:
    """Quad -> EasyOCR horizontal box format [x_min, x_max, y_min, y_max]."""
    x0, y0, x1, y1 = _quad_to_box(pts)
    return int(x0), int(x1), int(y0), int(y1)


def _dedupe(lines: list[TextLine]) -> list[TextLine]:
    """Drop duplicates produced by overlapping tiles, keeping the higher-confidence read."""
    kept: list[TextLine] = []
    for line in sorted(lines, key=lambda l: -l.conf):
        if all(iou(line.bbox, k.bbox) < 0.5 for k in kept):
            kept.append(line)
    return kept


class OCRBackend:
    max_side = 3600  # larger sheets are tiled (a 200-dpi A3 sheet fits in one pass)

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

    def _recognize(self, img: np.ndarray, boxes: list) -> list[tuple[str, float]]:
        res = self.reader.recognize(img, horizontal_list=boxes, free_list=[], detail=1, paragraph=False,
                                    batch_size=32)
        return [(text, float(conf)) for _, text, conf in res]

    def _read(self, img: np.ndarray) -> list[TextLine]:
        # Detect once (CRAFT), then recognize horizontal boxes in one batch. Tall boxes are
        # vertical annotations: rotate the crop both ways and keep the more confident read.
        # (readtext(rotation_info=...) would re-recognize *every* box 3x, ~3x slower on CPU.)
        horiz, free = self.reader.detect(img, canvas_size=max(img.shape[:2]), mag_ratio=1.0,
                                         text_threshold=0.6, low_text=0.35, link_threshold=0.4)
        horiz, free = horiz[0], free[0]
        flat = [b for b in horiz if (b[3] - b[2]) <= 1.5 * (b[1] - b[0])]
        flat += [[*_quad_bounds(q)] for q in free]  # slightly skewed quads -> axis-aligned boxes
        tall = [b for b in horiz if (b[3] - b[2]) > 1.5 * (b[1] - b[0])]

        out = []
        if flat:
            for b, (text, conf) in zip(flat, self._recognize(img, flat)):
                out.append(TextLine(text, tuple(float(v) for v in (b[0], b[2], b[1], b[3])), conf, "ocr"))
        for x0, x1, y0, y1 in tall:
            crop = img[max(0, y0):y1, max(0, x0):x1]
            best = ("", 0.0)
            for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                r = cv2.rotate(crop, rot)
                reads = self._recognize(r, [[0, r.shape[1], 0, r.shape[0]]])
                if reads and reads[0][1] > best[1]:
                    best = reads[0]
            out.append(TextLine(best[0], tuple(float(v) for v in (x0, y0, x1, y1)), best[1], "ocr"))
        return [l for l in out if l.text.strip()]


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
