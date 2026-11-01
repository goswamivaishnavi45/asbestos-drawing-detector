"""Evaluate the pipeline on the synthetic set.

Reports, per input mode (vector PDF / degraded scan):
  - text detection P/R/F1 (IoU >= 0.5), CER (all / Japanese / English lines), WER (English lines)
  - construction-year extraction accuracy
  - asbestos flagging P/R/F1 (flag = ASBESTOS or NEEDS_REVIEW), 3-class accuracy, material-ID accuracy
  - matcher ablation: exact vs fuzzy vs tfidf vs hybrid on clean and OCR'd text

The match threshold is chosen on the dev split (first half) and reported on the test split.
OCR output is cached in results/cache so re-running metrics is fast.

    python scripts/evaluate.py --data data/synthetic --modes vector scan
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asbestos_detector.database import MaterialDB  # noqa: E402
from asbestos_detector.geometry import TextLine  # noqa: E402
from asbestos_detector.ingest import load_pages  # noqa: E402
from asbestos_detector.matcher import METHODS, MaterialMatcher  # noqa: E402
from asbestos_detector.metrics import cer, match_boxes, prf, wer  # noqa: E402
from asbestos_detector.normalize import parse_annotation  # noqa: E402
from asbestos_detector.pipeline import Analyzer, find_build_year  # noqa: E402
from asbestos_detector.preprocess import preprocess  # noqa: E402
from asbestos_detector.rules import ASBESTOS, REVIEW  # noqa: E402
from asbestos_detector.visualize import annotate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FLAG = {ASBESTOS, REVIEW}
THRESHOLDS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def read_cached(analyzer: Analyzer, gt: dict, data: Path, mode: str, cache_dir: Path, tag: str):
    """Text lines + table cells for one drawing, OCR'd once and cached."""
    cache = cache_dir / f"{gt['id']}_{mode}_{tag}.json"
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        return [TextLine(l["text"], tuple(l["bbox"]), l["conf"], l["source"]) for l in c["lines"]], c["cells"], c["angle"], c["seconds"]
    if mode == "vector":
        page = load_pages(data / f"{gt['id']}.pdf", dpi=150)[0]
    else:
        page = load_pages(data / f"{gt['id']}_scan.png", dpi=gt["scan_dpi"])[0]
    t0 = time.time()
    lines, _, angle, tables, _ = analyzer.read_page_lines(page, force_ocr=(mode == "scan"))
    secs = time.time() - t0
    cells = [list(c) for t in tables for c in t.cells]
    cache.write_text(json.dumps({"lines": [l.__dict__ for l in lines], "cells": cells, "angle": angle, "seconds": secs},
                                ensure_ascii=False), encoding="utf-8")
    return lines, cells, angle, secs


def gt_boxes(gt: dict, mode: str) -> list[tuple]:
    s = (150 if mode == "vector" else gt["scan_dpi"]) / 72
    return [tuple(v * s for v in it["bbox_pt"]) for it in gt["items"]]


def score_page(gt, lines, cells, analyzer, mode) -> Counter:
    c = Counter()
    gboxes = gt_boxes(gt, mode)
    items = gt["items"]
    year = find_build_year(lines)
    c["year_ok"] += int(year == gt["build_year"])
    c["pages"] += 1

    # text detection + recognition
    det = match_boxes(gboxes, [l.bbox for l in lines], 0.5)
    c["det_tp"] += len(det)
    c["det_fp"] += len(lines) - len(det)
    c["det_fn"] += len(gboxes) - len(det)
    pairs = dict(match_boxes(gboxes, [l.bbox for l in lines], 0.3))
    for gi, it in enumerate(items):
        hyp = lines[pairs[gi]].text if gi in pairs else ""
        d, n = cer(it["text"], hyp)
        c["cer_d"] += d
        c["cer_n"] += n
        c[f"cer_d_{it['lang']}"] += d
        c[f"cer_n_{it['lang']}"] += n
        if it["lang"] == "en":
            d, n = wer(it["text"], hyp)
            c["wer_d"] += d
            c["wer_n"] += n

    # classification (end to end, using the year the pipeline itself extracted)
    pred = analyzer.classify_lines(lines, year, 0, [tuple(x) for x in cells])
    for gi, it in enumerate(items):
        p = pred[pairs[gi]] if gi in pairs else None
        gt_flag = it.get("expected_label") in FLAG
        pred_flag = bool(p and p.label in FLAG)
        if it["kind"] == "material":
            c["mat_n"] += 1
            c["mat_label_ok"] += int(bool(p) and p.label == it["expected_label"])
            c["mat_id_ok"] += int(bool(p) and p.material_id == it["entry_id"])
            c["mat_detected"] += int(bool(p and p.label))
        elif p and p.label:
            c["nonmat_false_material"] += 1
        c["flag_tp"] += int(gt_flag and pred_flag)
        c["flag_fn"] += int(gt_flag and not pred_flag)
        c["flag_fp"] += int(pred_flag and not gt_flag)
    matched_preds = set(pairs.values())
    c["flag_fp"] += sum(1 for i, p in enumerate(pred) if i not in matched_preds and p.label in FLAG)
    return c


def matcher_ablation(gt, lines, matchers, mode) -> Counter:
    c = Counter()
    pairs = dict(match_boxes(gt_boxes(gt, mode), [l.bbox for l in lines], 0.3))
    for gi, it in enumerate(gt["items"]):
        ocr_text = lines[pairs[gi]].text if gi in pairs else ""
        for name, m in matchers.items():
            for src, text in (("clean", it["text"]), ("ocr", ocr_text)):
                hit = m.match(parse_annotation(text).body) if text else None
                if it["kind"] == "material":
                    c[f"{name}_{src}_n"] += 1
                    c[f"{name}_{src}_ok"] += int(bool(hit) and hit.entry.id == it["entry_id"])
                else:
                    c[f"{name}_{src}_neg"] += 1
                    c[f"{name}_{src}_fm"] += int(bool(hit))
    return c


def save_example(analyzer: Analyzer, gt: dict, data: Path, mode: str, read: dict, tag: str) -> None:
    """Annotated image of one test drawing with the final threshold (re-runs preprocessing only, not OCR)."""
    lines, cells = read[gt["id"]]
    if mode == "vector":
        img = load_pages(data / f"{gt['id']}.pdf", dpi=150)[0].image
    else:
        page = load_pages(data / f"{gt['id']}_scan.png", dpi=gt["scan_dpi"])[0]
        img, _ = preprocess(page.image, analyzer.deskew, analyzer.denoise)
    items = analyzer.classify_lines(lines, find_build_year(lines), 0, [tuple(x) for x in cells])
    vis = annotate(img, items)
    scale = 1600 / vis.shape[1]
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imencode(".png", vis)[1].tofile(str(ROOT / "docs" / f"example_{mode}_{tag}.png"))


def summarize(c: Counter) -> dict:
    r = lambda a, b: round(c[a] / c[b], 4) if c[b] else None
    return {
        "pages": c["pages"],
        "detection": prf(c["det_tp"], c["det_fp"], c["det_fn"]),
        "cer": r("cer_d", "cer_n"), "cer_ja": r("cer_d_ja", "cer_n_ja"), "cer_en": r("cer_d_en", "cer_n_en"),
        "wer_en": r("wer_d", "wer_n"),
        "year_accuracy": r("year_ok", "pages"),
        "materials": c["mat_n"],
        "material_detected_rate": r("mat_detected", "mat_n"),
        "material_id_accuracy": r("mat_id_ok", "mat_n"),
        "label_accuracy_3class": r("mat_label_ok", "mat_n"),
        "asbestos_flag": prf(c["flag_tp"], c["flag_fp"], c["flag_fn"]),
        "non_material_false_matches": c["nonmat_false_material"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--modes", nargs="+", default=["vector", "scan"])
    ap.add_argument("--ocr", default="easyocr")
    ap.add_argument("--no-preprocess", action="store_true", help="ablation: skip deskew + denoise")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    data, out = ROOT / args.data, ROOT / args.out
    tag = "raw" if args.no_preprocess else "pre"
    (out / "cache").mkdir(parents=True, exist_ok=True)
    gts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(data.glob("drawing_*.json"))][: args.limit]
    half = len(gts) // 2
    splits = {"dev": gts[:half], "test": gts[half:]}

    db = MaterialDB.load()
    analyzer = Analyzer(db, ocr=args.ocr, deskew=not args.no_preprocess, denoise=not args.no_preprocess)
    results = {"n_drawings": len(gts), "split": {"dev": half, "test": len(gts) - half}, "ocr": args.ocr,
               "preprocess": not args.no_preprocess, "modes": {}}

    for mode in args.modes:
        print(f"== {mode}")
        read = {}
        secs, skew_err = [], []
        for i, gt in enumerate(gts):
            lines, cells, angle, s = read_cached(analyzer, gt, data, mode, out / "cache", tag)
            read[gt["id"]] = (lines, cells)
            secs.append(s)
            if mode == "scan" and not args.no_preprocess:
                skew_err.append(abs(angle + gt["scan_skew_deg"]))  # correction should cancel the applied skew
            print(f"  {gt['id']}: {len(lines)} lines, skew {angle:+.2f} (true {gt['scan_skew_deg'] if mode == 'scan' else 0:+.2f}), {s:.1f}s")

        # pick the threshold on dev by asbestos-flag F1 (ties -> higher recall, then the stricter threshold)
        sweep = {}
        for t in THRESHOLDS:
            analyzer.matcher.threshold = t
            c = sum((score_page(g, *read[g["id"]], analyzer, mode) for g in splits["dev"]), Counter())
            sweep[t] = summarize(c)["asbestos_flag"]
        best_t = max(THRESHOLDS, key=lambda t: (sweep[t]["f1"], sweep[t]["recall"], t))
        analyzer.matcher.threshold = best_t
        save_example(analyzer, splits["test"][0], data, mode, read, tag)

        mode_res = {"threshold": best_t, "dev_threshold_sweep": sweep, "avg_seconds_per_page": round(sum(secs) / len(secs), 2)}
        if skew_err:
            mode_res["deskew_mean_abs_error_deg"] = round(sum(skew_err) / len(skew_err), 4)
        for split, gs in splits.items():
            mode_res[split] = summarize(sum((score_page(g, *read[g["id"]], analyzer, mode) for g in gs), Counter()))

        matchers = {m: MaterialMatcher(db, m, best_t if m != "exact" else 0.5) for m in METHODS}
        ab = sum((matcher_ablation(g, read[g["id"]][0], matchers, mode) for g in splits["test"]), Counter())
        mode_res["matcher_ablation_test"] = {
            m: {f"{src}_material_id_acc": round(ab[f"{m}_{src}_ok"] / ab[f"{m}_{src}_n"], 4) for src in ("clean", "ocr")}
               | {"ocr_false_match_rate_on_non_materials": round(ab[f"{m}_ocr_fm"] / ab[f"{m}_ocr_neg"], 4)}
            for m in METHODS
        }
        results["modes"][mode] = mode_res
        print(json.dumps(mode_res["test"], indent=1, ensure_ascii=False))

    path = out / f"metrics_{tag}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
