"""Analyze one drawing and write a CSV/JSON report plus an annotated image.

    python scripts/analyze.py data/samples/drawing_000_scan.png --out out/
    python scripts/analyze.py drawing.pdf --year 1978
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asbestos_detector import Analyzer  # noqa: E402
from asbestos_detector.visualize import annotate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drawing", help="PDF (vector or scanned) or image")
    ap.add_argument("--out", default="out")
    ap.add_argument("--year", type=int, help="construction year, if not in the title block")
    ap.add_argument("--ocr", default="easyocr", choices=["easyocr", "paddleocr"])
    ap.add_argument("--force-ocr", action="store_true", help="OCR even when the PDF has embedded text")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report, images = Analyzer(ocr=args.ocr).analyze(args.drawing, args.year, args.force_ocr)
    stem = Path(args.drawing).stem
    report.save_csv(out / f"{stem}_report.csv")
    report.save_json(out / f"{stem}_report.json")
    for page, img in zip(report.pages, images):
        vis = annotate(img, [i for i in report.items if i.page == page.index])
        cv2.imencode(".png", vis)[1].tofile(str(out / f"{stem}_p{page.index}_annotated.png"))

    print(f"construction year: {report.build_year} ({report.build_year_source})")
    for i, it in enumerate(report.materials, 1):
        print(f"#{i:<3} {it.label:13s} {it.text:40s} -> {it.material} ({it.match_score})")
    print(f"\nreport written to {out}/")


if __name__ == "__main__":
    main()
