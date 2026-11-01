"""Streamlit demo:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asbestos_detector import ASBESTOS, NON_ASBESTOS, REVIEW, Analyzer  # noqa: E402
from asbestos_detector.visualize import annotate  # noqa: E402

st.set_page_config(page_title="Asbestos Drawing Detector", layout="wide")


@st.cache_resource
def get_analyzer() -> Analyzer:
    return Analyzer()


st.title("Asbestos material detection from building drawings")
st.caption("Upload a vector or scanned drawing (PDF / PNG / JPG). Text is extracted, matched against the "
           "asbestos building-materials database and classified. Screening aid only — confirm with sampling and lab analysis.")

samples = sorted((ROOT / "data" / "samples").glob("*"))
col1, col2 = st.columns([2, 1])
upload = col1.file_uploader("Drawing", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
sample = col2.selectbox("…or try a sample", ["—"] + [p.name for p in samples if p.suffix in {".pdf", ".png"}])
year = col2.number_input("Construction year (0 = read from title block)", 0, 2030, 0)
force_ocr = col2.checkbox("Force OCR on vector PDFs")

path = None
if upload:
    tmp = Path(tempfile.mkdtemp()) / upload.name
    tmp.write_bytes(upload.getvalue())
    path = tmp
elif sample != "—":
    path = ROOT / "data" / "samples" / sample

if path:
    with st.spinner("Analyzing… (first OCR run downloads model weights)"):
        report, images = get_analyzer().analyze(path, year or None, force_ocr)
    mats = report.materials
    c = st.columns(4)
    c[0].metric("Construction year", report.build_year or "unknown", report.build_year_source)
    c[1].metric("Asbestos-containing", sum(i.label == ASBESTOS for i in mats))
    c[2].metric("Needs review", sum(i.label == REVIEW for i in mats))
    c[3].metric("Not containing", sum(i.label == NON_ASBESTOS for i in mats))
    for page, img in zip(report.pages, images):
        vis = annotate(img, [i for i in report.items if i.page == page.index])
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                 caption=f"Page {page.index + 1} · {page.mode} · skew corrected {page.skew_angle}° · tables {page.n_tables}  "
                         "(red = asbestos, orange = review, green = not containing)")
    df = pd.DataFrame([{"#": n, "label": i.label, "text": i.text, "material": i.material, "score": i.match_score,
                        "location": i.location, "region": i.region, "reason": i.reason} for n, i in enumerate(mats, 1)])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8-sig"), "asbestos_report.csv", "text/csv")
