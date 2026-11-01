"""OCR and classification metrics: CER, WER, IoU-matched detection P/R/F1."""
from __future__ import annotations

import re

from rapidfuzz.distance import Levenshtein

from .geometry import Box, iou
from .normalize import nfkc


def _clean(s: str) -> str:
    # NFKC + drop whitespace: OCR engines are inconsistent about spaces between CJK characters.
    return re.sub(r"\s+", "", nfkc(s))


def cer(ref: str, hyp: str) -> tuple[int, int]:
    """Returns (edit distance, reference length) so corpus CER = sum(d) / sum(n)."""
    r, h = _clean(ref), _clean(hyp)
    return Levenshtein.distance(r, h), len(r)


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """Word-level edit distance over whitespace tokens (meaningful for English lines)."""
    r, h = nfkc(ref).lower().split(), nfkc(hyp).lower().split()
    return Levenshtein.distance(r, h), len(r)


def match_boxes(gt: list[Box], pred: list[Box], thr: float = 0.5) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by IoU. Returns (gt_idx, pred_idx) pairs."""
    pairs = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)), reverse=True)
    used_g, used_p, out = set(), set(), []
    for score, gi, pi in pairs:
        if score < thr:
            break
        if gi not in used_g and pi not in used_p:
            used_g.add(gi), used_p.add(pi), out.append((gi, pi))
    return out


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}
