"""Draw classification results onto the drawing."""
from __future__ import annotations

import cv2
import numpy as np

from .pipeline import Item
from .rules import ASBESTOS, NON_ASBESTOS, REVIEW

COLORS = {ASBESTOS: (40, 40, 220), REVIEW: (0, 150, 255), NON_ASBESTOS: (60, 170, 60)}  # BGR
OTHER = (190, 190, 190)


def annotate(gray: np.ndarray, items: list[Item], show_other: bool = True) -> np.ndarray:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    n = 0
    for it in items:
        x0, y0, x1, y1 = (int(round(v)) for v in it.bbox)
        if it.label:
            n += 1
            color = COLORS[it.label]
            cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), color, 3)
            cv2.putText(img, f"#{n}", (x0, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        elif show_other:
            cv2.rectangle(img, (x0, y0), (x1, y1), OTHER, 1)
    return img
