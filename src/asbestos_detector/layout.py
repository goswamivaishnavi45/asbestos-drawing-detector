"""Table detection for finish schedules (仕上表) using morphological ruling-line extraction."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import Box


@dataclass
class Table:
    bbox: Box
    cells: list[Box] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0


def _cluster(values: list[float], tol: float) -> int:
    count, last = 0, None
    for v in sorted(values):
        if last is None or v - last > tol:
            count += 1
        last = v
    return count


def detect_tables(gray: np.ndarray, min_cells: int = 6) -> list[Table]:
    h, w = gray.shape[:2]
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 50), 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 50))))
    grid = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8))

    # Cells are the enclosed background regions of the ruling-line grid.
    n, _, stats, _ = cv2.connectedComponentsWithStats(255 - grid, connectivity=4)
    cells: list[Box] = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if x == 0 or y == 0 or x + cw >= w or y + ch >= h:
            continue  # touches image border: not enclosed
        if ch > 0.08 * h or cw > 0.4 * w or cw < 15 or ch < 8:
            continue  # rooms in the plan are much taller than schedule rows
        if area < 0.85 * cw * ch:
            continue  # not rectangular
        cells.append((float(x), float(y), float(x + cw), float(y + ch)))

    # Group adjacent cells into tables (union-find on touching boxes).
    parent = list(range(len(cells)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pad = 8
    for i, a in enumerate(cells):
        for j in range(i + 1, len(cells)):
            b = cells[j]
            if a[0] - pad <= b[2] and b[0] - pad <= a[2] and a[1] - pad <= b[3] and b[1] - pad <= a[3]:
                parent[find(i)] = find(j)

    groups: dict[int, list[Box]] = {}
    for i, c in enumerate(cells):
        groups.setdefault(find(i), []).append(c)

    tables = []
    for group in groups.values():
        rows = _cluster([c[1] for c in group], tol=6)
        cols = _cluster([c[0] for c in group], tol=6)
        if len(group) < min_cells or rows < 2 or cols < 2:
            continue
        bbox = (min(c[0] for c in group), min(c[1] for c in group), max(c[2] for c in group), max(c[3] for c in group))
        tables.append(Table(bbox, sorted(group, key=lambda c: (c[1], c[0])), rows, cols))
    return tables
