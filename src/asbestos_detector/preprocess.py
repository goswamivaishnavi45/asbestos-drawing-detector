"""Image preprocessing for scanned drawings: deskew, denoise, binarize, tile."""
from __future__ import annotations

import math

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0) -> float:
    """Skew angle in degrees, estimated from the long straight lines every drawing has (borders, walls, tables)."""
    edges = cv2.Canny(gray, 50, 150)
    min_len = max(gray.shape) // 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=150, minLineLength=min_len, maxLineGap=5)
    if lines is None:
        return 0.0
    angles = []
    for x0, y0, x1, y1 in lines.reshape(-1, 4):  # OpenCV 4 returns (N,1,4), OpenCV 5 (N,4)
        a = math.degrees(math.atan2(y1 - y0, x1 - x0))
        if abs(a) < max_angle:
            angles.append(a)
        elif abs(abs(a) - 90) < max_angle:  # near-vertical line
            angles.append(a - 90 if a > 0 else a + 90)
    coarse = float(np.median(angles)) if angles else 0.0
    return _refine_skew(gray, coarse)


def _refine_skew(gray: np.ndarray, coarse: float, span: float = 0.6, step: float = 0.02) -> float:
    """Projection-profile search around the Hough estimate: the right angle makes ink rows sharpest."""
    scale = 1200 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = ink.shape
    best, best_score = coarse, -1.0
    for a in np.arange(coarse - span, coarse + span + 1e-9, step):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1.0)
        rows = cv2.warpAffine(ink, m, (w, h)).sum(axis=1, dtype=np.float64)
        score = float(np.sum(np.diff(rows) ** 2))
        if score > best_score:
            best, best_score = float(a), score
    return best


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=15)


def binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def preprocess(img: np.ndarray, do_deskew: bool = True, do_denoise: bool = True) -> tuple[np.ndarray, float]:
    """Returns the cleaned grayscale image and the skew angle that was corrected."""
    gray = to_gray(img)
    angle = 0.0
    if do_deskew:
        angle = estimate_skew(gray)
        if abs(angle) > 0.05:
            gray = rotate(gray, angle)
    if do_denoise:
        gray = denoise(gray)
    return gray, angle


def tiles(shape: tuple[int, int], tile: int = 1600, overlap: int = 200):
    """Yields (x, y, w, h) tiles covering an image, for OCR on large drawing sheets."""
    h, w = shape[:2]
    step = tile - overlap
    for y in range(0, max(h - overlap, 1), step):
        for x in range(0, max(w - overlap, 1), step):
            yield x, y, min(tile, w - x), min(tile, h - y)
