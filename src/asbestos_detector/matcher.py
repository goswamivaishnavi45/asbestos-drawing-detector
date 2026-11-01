"""Match OCR'd material phrases against the materials database.

Methods (compared in scripts/evaluate.py):
  exact  - dictionary lookup on normalized keys (exact or substring)
  fuzzy  - RapidFuzz edit-distance similarity
  tfidf  - character n-gram TF-IDF cosine similarity (robust to OCR character swaps)
  hybrid - fuzzy + tfidf rankings fused with Reciprocal Rank Fusion (RRF)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from .database import MaterialDB, MaterialEntry
from .normalize import normalize_key

METHODS = ("exact", "fuzzy", "tfidf", "hybrid")
RRF_K = 60


@dataclass
class Match:
    entry: MaterialEntry
    score: float  # 0..1 confidence
    matched_form: str
    method: str


class MaterialMatcher:
    def __init__(self, db: MaterialDB, method: str = "hybrid", threshold: float = 0.75):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.db, self.method, self.threshold = db, method, threshold
        self.forms: list[tuple[str, str, int]] = []  # (key, original form, entry index)
        for i, e in enumerate(db.entries):
            for form in e.surface_forms:
                key = normalize_key(form)
                if key:
                    self.forms.append((key, form, i))
        self.keys = [k for k, _, _ in self.forms]
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)
        self.form_vecs = self.vectorizer.fit_transform(self.keys)

    # --- per-form scores -------------------------------------------------------
    def _exact(self, q: str) -> np.ndarray:
        return np.array([1.0 if q == k or (len(k) >= 4 and k in q) else 0.0 for k in self.keys])

    def _fuzzy(self, q: str) -> np.ndarray:
        # Partial matching (WRatio) only when the query is at least as long as most of the form:
        # "天井:岩綿吸音板 t=12" should hit "岩綿吸音板", but a fragment like "floor" must not hit
        # "vinyl floor tile". Short keys (PB, RC, ECP) also need whole-string similarity.
        return np.array([
            (fuzz.ratio(q, k) if len(k) <= 3 or len(q) < 0.6 * len(k) else fuzz.WRatio(q, k, processor=None)) / 100
            for k in self.keys
        ])

    def _tfidf(self, q: str) -> np.ndarray:
        return (self.form_vecs @ self.vectorizer.transform([q]).T).toarray().ravel()

    def _per_entry(self, form_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Best score per entry and the index of the form that produced it."""
        best = np.full(len(self.db), -1.0)
        arg = np.zeros(len(self.db), dtype=int)
        for fi, (_, _, ei) in enumerate(self.forms):
            if form_scores[fi] > best[ei]:
                best[ei], arg[ei] = form_scores[fi], fi
        return best, arg

    # --- public API ----------------------------------------------------------------
    def candidates(self, text: str, top_k: int = 3) -> list[Match]:
        q = normalize_key(text)
        if not q:
            return []
        if self.method == "exact":
            best, arg = self._per_entry(self._exact(q))
            order, conf = np.argsort(-best), best
        elif self.method == "fuzzy":
            best, arg = self._per_entry(self._fuzzy(q))
            order, conf = np.argsort(-best), best
        elif self.method == "tfidf":
            best, arg = self._per_entry(self._tfidf(q))
            order, conf = np.argsort(-best), best
        else:
            exact_best, exact_arg = self._per_entry(self._exact(q))
            fz, fz_arg = self._per_entry(self._fuzzy(q))
            tf, tf_arg = self._per_entry(self._tfidf(q))
            rank_fz = np.argsort(np.argsort(-fz))
            rank_tf = np.argsort(np.argsort(-tf))
            rrf = 1 / (RRF_K + rank_fz + 1) + 1 / (RRF_K + rank_tf + 1) + exact_best  # exact hit wins outright
            order = np.argsort(-rrf)
            conf = np.maximum(np.maximum(fz, tf), exact_best)
            arg = np.where(exact_best > 0, exact_arg, np.where(fz >= tf, fz_arg, tf_arg))
        return [
            Match(self.db.entries[i], float(conf[i]), self.forms[arg[i]][1], self.method)
            for i in order[:top_k]
        ]

    def match(self, text: str) -> Match | None:
        cands = self.candidates(text, top_k=1)
        if cands and cands[0].score >= self.threshold:
            return cands[0]
        return None
