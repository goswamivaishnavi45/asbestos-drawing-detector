"""REST API.

    uvicorn app.api:app --reload
    curl -F "file=@drawing.pdf" -F "build_year=1978" localhost:8000/analyze
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asbestos_detector import Analyzer  # noqa: E402
from asbestos_detector.ingest import IMAGE_EXTS  # noqa: E402

app = FastAPI(title="Asbestos Drawing Detector", version="0.1.0")
analyzer = Analyzer()
MAX_BYTES = 50 * 1024 * 1024


@app.get("/health")
def health():
    return {"status": "ok", "materials_in_db": len(analyzer.db)}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), build_year: int | None = Form(None), force_ocr: bool = Form(False)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".pdf" and suffix not in IMAGE_EXTS:
        raise HTTPException(415, f"unsupported file type {suffix!r}")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        report, _ = analyzer.analyze(path, build_year, force_ocr)
    report.file = file.filename or report.file
    summary = {}
    for it in report.materials:
        summary[it.label] = summary.get(it.label, 0) + 1
    return {"summary": summary, **report.to_dict(), "items": [i.__dict__ for i in report.materials]}
