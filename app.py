"""
Product Catalog + PG Classifier (MODEL-ONLY)

- Upload PDF (catalog) + Excel (SKUs)
- Extract product images from PDF
- (Optional) OCR each image and match to SKU rows (else sequential)
- Load a cached scikit-learn PG classifier from disk
- Predict PG per row from Vendor Category + Product Title
- Let the user edit predictions; export Excel with images + PG_final
- Sidebar button to batch retrain model from /training_data via retrain_pg_model.py
"""

import os
import sys
import time
import subprocess
import tempfile
from io import BytesIO
from typing import List, Tuple

import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
from scipy import ndimage

from openpyxl.drawing.image import Image as XLImage
import openpyxl.utils

# Optional OCR / fuzzy (gracefully degrade if absent)
try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None  # type: ignore
try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:
    fuzz = None  # type: ignore

import streamlit as st
import joblib


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING / RETRAIN
# ──────────────────────────────────────────────────────────────────────────────

PG_MODEL_CANDIDATES = [
    "models/pg_classifier_latest.joblib",
    "models/pg_classifier_final.joblib",   # fallback
    "./pg_classifier_latest.joblib",
    "./pg_classifier_final.joblib",
]

@st.cache_resource(show_spinner="Loading PG model…")
def load_pg_model():
    tried = []
    for p in PG_MODEL_CANDIDATES:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception as e:
                tried.append(f"{p} (failed: {e})")
        else:
            tried.append(f"{p} (missing)")
    st.warning("PG model not found. Tried: " + " | ".join(tried))
    return None


# ──────────────────────────────────────────────────────────────────────────────
# PDF → IMAGE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_images_from_pdf(
    pdf_path: str,
    threshold: int = 200,
    min_area_ratio: float = 0.02,
    max_images_per_page: int = 1,
) -> List[Image.Image]:
    """Convert PDF pages into images and crop large non‑white regions."""
    out: List[Image.Image] = []
    pages = convert_from_path(pdf_path, dpi=200)
    for page in pages:
        scale = 4
        small = page.resize((max(1, page.width // scale), max(1, page.height // scale)))
        arr = np.array(small.convert("L"))
        binary = arr < threshold
        labeled, num = ndimage.label(binary)
        if num == 0:
            continue
        objects = ndimage.find_objects(labeled)
        page_area = small.width * small.height
        bboxes: List[Tuple[int, Tuple[int, int, int, int]]] = []
        for slc in objects:
            y0, y1 = slc[0].start, slc[0].stop
            x0, x1 = slc[1].start, slc[1].stop
            area = (x1 - x0) * (y1 - y0)
            if page_area and (area / page_area) >= min_area_ratio:
                bboxes.append((area, (x0, y0, x1, y1)))
        if not bboxes:
            continue
        bboxes.sort(reverse=True, key=lambda t: t[0])
        selected = bboxes[:max_images_per_page]
        sx = page.width / small.width
        sy = page.height / small.height
        for _, (x0, y0, x1, y1) in selected:
            left, top, right, bottom = int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)
            left = max(left, 0); top = max(top, 0)
            right = min(right, page.width); bottom = min(bottom, page.height)
            out.append(page.crop((left, top, right, bottom)))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# OCR + MATCHING
# ──────────────────────────────────────────────────────────────────────────────

def ocr_extract_texts(images: List[Image.Image]) -> List[str]:
    """OCR each image; return empty strings if OCR unavailable or fails."""
    if pytesseract is None:
        return ["" for _ in images]
    texts = []
    for img in images:
        try:
            t = pytesseract.image_to_string(img.convert("L"))
            texts.append(" ".join(t.strip().split()))
        except Exception:
            texts.append("")
    return texts


def match_images_to_skus(texts: List[str], df: pd.DataFrame, match_column: str) -> List[int]:
    """Return per-image row indices (unique). Falls back to sequential if weak."""
    n = len(df)
    available = list(range(n))
    results: List[int] = []
    lowers = [str(df.iloc[i][match_column]).lower() if i < n else "" for i in range(n)]
    for t in texts:
        if not available:
            results.append(-1); continue
        tl = t.lower().strip()
        best, best_score = available[0], -1.0
        for idx in available:
            cand = lowers[idx]
            if fuzz is not None:
                try:
                    score = fuzz.partial_ratio(tl, cand)
                except Exception:
                    score = 0.0
            else:
                score = 100.0 if tl and tl in cand else 0.0
            if score > best_score:
                best, best_score = idx, score
        if fuzz is not None and best_score < 30:
            best = available[0]  # weak match → sequential fallback
        results.append(best)
        available.remove(best)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# EXCEL OUTPUT WITH EMBEDDED IMAGES
# ──────────────────────────────────────────────────────────────────────────────

def embed_images_to_excel(
    df: pd.DataFrame,
    images: List[Image.Image],
    row_indices: List[int],
    output_stream: BytesIO,
    image_column: str = "Image",
    pg_column: str = "PG_final",
) -> None:
    """Write df to Excel, embed images in image_column, include PG_final."""
    if len(row_indices) != len(images):
        raise ValueError("row_indices length must equal number of images.")
    n = len(df)
    if row_indices and max(row_indices) >= n:
        raise ValueError("row_indices contains an out-of-range index.")

    df_out = df.copy().reset_index(drop=True)
    if image_column not in df_out.columns:
        df_out[image_column] = ""
    if pg_column not in df_out.columns:
        df_out[pg_column] = ""

    tmp_files: List[str] = []
    try:
        with pd.ExcelWriter(output_stream, engine="openpyxl") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Sheet1")
            ws = writer.sheets["Sheet1"]
            img_col_idx = df_out.columns.get_loc(image_column) + 1
            col_letter = openpyxl.utils.get_column_letter(img_col_idx)
            ws.column_dimensions[col_letter].width = 25

            import tempfile as _tmp
            for i, img in enumerate(images):
                r = row_indices[i]
                t = _tmp.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(t, format="PNG")
                t.close()
                tmp_files.append(t.name)
                xl_img = XLImage(t.name)
                xl_img.width = 120; xl_img.height = 120
                ws.add_image(xl_img, f"{col_letter}{r + 2}")  # +2 = header + 1-index
    finally:
        for fn in tmp_files:
            try: os.remove(fn)
            except OSError: pass


# ──────────────────────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Normalize header: collapse whitespace/newlines and lowercase."""
    return " ".join(str(s).replace("\n", " ").split()).strip().lower()


def detect_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Find vendor category and product title columns via common aliases."""
    original = list(df.columns)
    CAND_VENDOR = [
        "vendor product category (required)", "vendor product category",
        "product category", "vendor category"
    ]
    CAND_TITLE = ["product title (required)", "product title", "title"]

    def by_alias(aliases):
        for a in aliases:
            for col in original:
                if _norm(col) == a:
                    return col
        return None

    vendor = by_alias(CAND_VENDOR)
    title = by_alias(CAND_TITLE)
    if not vendor:
        vendor = next((c for c in original if "product category" in _norm(c)), None)
    if not title:
        title = next((c for c in original if "title" in _norm(c)), None)
    return vendor, title


def main():
    st.set_page_config(page_title="Product Catalog + PG Classifier", layout="wide")
    st.title("Product Catalog + PG Classifier (model only)")

    # Sidebar: retrain
    with st.sidebar:
        st.markdown("### PG model")
        if st.button("🔄 Retrain from /training_data"):
            with st.spinner("Retraining – this may take a minute…"):
                res = subprocess.run(
                    [sys.executable, "retrain_pg_model.py",
                     "--input_folder", "training_data",
                     "--output_model", PG_MODEL_CANDIDATES[0]],
                    capture_output=True, text=True
                )
                st.code(res.stdout + "\n" + res.stderr)
                if res.returncode == 0:
                    load_pg_model.clear()
                    time.sleep(1)
                    st.success("Model retrained and cache cleared. New sessions will use it.")
                else:
                    st.error("Retraining failed – see log above.")

    # Advanced extraction
    with st.expander("Advanced PDF extraction"):
        threshold = st.slider("Greyscale threshold", 0, 255, 200, 5)
        min_area_ratio = st.number_input("Min region area ratio", 0.0, 1.0, 0.02, 0.01)
        max_images_per_page = st.number_input("Max images per page", 1, 10, 1, 1)

    pdf_file = st.file_uploader("Upload PDF catalog", type=["pdf"])
    excel_file = st.file_uploader("Upload SKU Excel", type=["xlsx", "xls", "xlsm"])

    use_ocr = st.checkbox(
        "Use OCR to match images to rows (else sequential order)", value=True,
        help="Requires pytesseract + system Tesseract. If unavailable, we fall back to sequential mapping."
    )

    if not pdf_file or not excel_file:
        st.info("Upload both a PDF and an Excel file to proceed.")
        return

    # Save uploads
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as fpdf:
        fpdf.write(pdf_file.getvalue()); pdf_path = fpdf.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as fxls:
        fxls.write(excel_file.getvalue()); excel_path = fxls.name

    # Extract images
    st.info("Extracting images from PDF…")
    try:
        images = extract_images_from_pdf(pdf_path, int(threshold), float(min_area_ratio), int(max_images_per_page))
    except Exception as exc:
        st.error(f"Failed to process PDF: {exc}"); return
    if not images:
        st.warning("No images extracted with current settings."); return
    st.success(f"Extracted {len(images)} image(s).")

    # Load model
    pg_model = load_pg_model()

    # Read Excel + Predict PG
    try:
        df_skus = pd.read_excel(excel_path)
        if len(df_skus) == 0:
            st.error("The SKU file is empty."); return

        vendor_cat_col, title_col = detect_cols(df_skus)
        st.info(f"Detected vendor category column: **{vendor_cat_col or '—'}**")
        st.info(f"Detected title column: **{title_col or '—'}**")

        if pg_model is not None and vendor_cat_col and title_col:
            text_series = df_skus[vendor_cat_col].astype(str) + " " + df_skus[title_col].astype(str)
            try:
                preds = pg_model.predict(text_series)
                confs = pg_model.predict_proba(text_series).max(axis=1)
                df_skus["PG_pred"] = preds
                df_skus["PG_confidence"] = np.round(confs, 3)
            except Exception as pred_exc:
                st.warning(f"PG prediction failed: {pred_exc}")
                df_skus["PG_pred"] = ""
                df_skus["PG_confidence"] = np.nan
        else:
            if pg_model is None:
                st.info("PG model not loaded – predictions unavailable.")
            else:
                st.info("Vendor Category / Product Title columns not found – predictions skipped.")
            if "PG_pred" not in df_skus.columns:
                df_skus["PG_pred"] = ""
            if "PG_confidence" not in df_skus.columns:
                df_skus["PG_confidence"] = np.nan

        df_skus["PG_final"] = df_skus["PG_pred"]

    except Exception as exc:
        st.error(f"Failed to read Excel file: {exc}")
        return

    # Row assignment (OCR or sequential)
    row_indices = list(range(len(images)))
    default_match_col = title_col or "Product Title (Required)"
    match_col = st.text_input(
        "SKU column to match OCR text against (e.g., 'Product Title (Required)')",
        value=default_match_col
    )
    if use_ocr and pytesseract is not None and fuzz is not None:
        st.info("Performing OCR and matching to rows…")
        texts = ocr_extract_texts(images)
        if match_col in df_skus.columns:
            row_indices = match_images_to_skus(texts, df_skus, match_col)
        else:
            st.warning(f"Match column '{match_col}' not found. Falling back to sequential assignment.")

    # Table: editable PG_pred
    st.subheader("PG suggestions (editable)")
    context_cols = []
    for c in df_skus.columns:
        if any(k in str(c).lower() for k in ["product category", "title", "style id"]):
            context_cols.append(c)
    cols = list(dict.fromkeys(context_cols + ["PG_pred", "PG_confidence"]))  # keep order, de-dupe
    edited = st.data_editor(
        df_skus[cols], num_rows="dynamic", use_container_width=True, key="editor"
    )
    if "PG_pred" in edited.columns:
        df_skus["PG_final"] = edited["PG_pred"]

    # Thumbnails with PG_final caption
    st.subheader("Sample extracted images")
    n_show = min(8, len(images))
    cols_ui = st.columns(n_show)
    for i in range(n_show):
        with cols_ui[i]:
            r = row_indices[i] if i < len(row_indices) else i
            label = ""
            if 0 <= r < len(df_skus):
                label = str(df_skus.iloc[r].get("PG_final", df_skus.iloc[r].get("PG_pred", "")))
            st.image(images[i].resize((180, 180)), caption=label if label else f"Image {i+1}")

    # Build downloadable Excel
    st.subheader("Download results")
    out = BytesIO()
    try:
        embed_images_to_excel(
            df=df_skus,
            images=images,
            row_indices=row_indices,
            output_stream=out,
            image_column="Image",
            pg_column="PG_final",
        )
    except Exception as exc:
        st.error(f"Failed to generate Excel: {exc}"); return
    out.seek(0)
    st.download_button(
        "Download updated Excel",
        data=out.getvalue(),
        file_name="catalog_with_pg.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
