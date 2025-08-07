"""
Product Catalog Processor + PG Classifier (model approach only)

What this does:
- Upload PDF (catalog) + Excel (SKUs)
- Extract product images from the PDF
- (Optional) OCR each image and match to SKU rows by a chosen text column
  – if OCR not available, assigns images to rows in order
- Load a cached scikit-learn Product Group (PG) classifier from disk
- Predict PG per row using Vendor Category + Product Title
- Let the user edit predictions; save edits to PG_final
- Embed images + PG_final into a downloadable Excel file
- Sidebar button to batch retrain model from /training_data using retrain_pg_model.py
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

# Excel embedding
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
import openpyxl.utils

# Optional OCR / fuzzy matching (will gracefully degrade if not present)
try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None  # type: ignore
try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:
    fuzz = None  # type: ignore

# Streamlit
import streamlit as st

# Model loading
import joblib


# ──────────────────────────────────────────────────────────────────────────────
# PG MODEL LOADING / RETRAIN
# ──────────────────────────────────────────────────────────────────────────────

PG_MODEL_PATH = "models/pg_classifier_latest.joblib"  # rename your file or path if needed

@st.cache_resource(show_spinner="Loading PG model…")
def load_pg_model(model_path: str = PG_MODEL_PATH):
    """Load the latest PG classifier from disk (shared across sessions)."""
    try:
        return joblib.load(model_path)
    except Exception:
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
    """Convert PDF pages into PIL images and crop large non‑white regions."""
    extracted: List[Image.Image] = []
    pages = convert_from_path(pdf_path, dpi=200)
    for page in pages:
        # Downsample for faster connected components
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
            left = max(left, 0)
            top = max(top, 0)
            right = min(right, page.width)
            bottom = min(bottom, page.height)
            crop = page.crop((left, top, right, bottom))
            extracted.append(crop)
    return extracted


# ──────────────────────────────────────────────────────────────────────────────
# OCR + MATCHING
# ──────────────────────────────────────────────────────────────────────────────

def ocr_extract_texts(images: List[Image.Image]) -> List[str]:
    """Return OCR text for each image or empty strings if OCR not available."""
    if pytesseract is None:
        return ["" for _ in images]
    out = []
    for img in images:
        try:
            text = pytesseract.image_to_string(img.convert("L"))
            text = " ".join(text.strip().split())
        except Exception:
            text = ""
        out.append(text)
    return out


def match_images_to_skus(texts: List[str], df: pd.DataFrame, match_column: str) -> List[int]:
    """
    Map each image text to the best SKU row index, uniquely.
    If fuzzy lib unavailable or match weak, falls back to next available row.
    """
    n = len(df)
    available = list(range(n))
    results: List[int] = []
    lower_vals = [str(df.iloc[i][match_column]).lower() if i < n else "" for i in range(n)]
    for t in texts:
        if not available:
            results.append(-1)
            continue
        tl = t.lower().strip()
        best, best_score = available[0], -1.0
        for idx in available:
            cand = lower_vals[idx]
            if fuzz is not None:
                try:
                    score = fuzz.partial_ratio(tl, cand)
                except Exception:
                    score = 0.0
            else:
                score = 100.0 if tl and tl in cand else 0.0
            if score > best_score:
                best, best_score = idx, score
        # If fuzzy score too weak, just assign next sequential row
        if fuzz is not None and best_score < 30:
            best = available[0]
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
    """Write df to Excel, embed images to matched rows, and include PG_final."""
    n_rows = df.shape[0]
    if len(row_indices) != len(images):
        raise ValueError("row_indices length must match number of images.")
    if row_indices and max(row_indices) >= n_rows:
        raise ValueError("row_indices contains an out-of-range index.")

    df_out = df.copy().reset_index(drop=True)
    if image_column not in df_out.columns:
        df_out[image_column] = ""
    if pg_column not in df_out.columns:
        df_out[pg_column] = ""

    temp_files: List[str] = []
    try:
        with pd.ExcelWriter(output_stream, engine="openpyxl") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Sheet1")
            wb = writer.book
            ws = writer.sheets["Sheet1"]

            # Make image column wide
            img_col_idx = df_out.columns.get_loc(image_column) + 1
            col_letter = openpyxl.utils.get_column_letter(img_col_idx)
            ws.column_dimensions[col_letter].width = 25

            import tempfile as _tmp
            for i, img in enumerate(images):
                row_idx = row_indices[i]
                # save temporary PNG
                t = _tmp.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(t, format="PNG")
                t.close()
                temp_files.append(t.name)

                xl_img = XLImage(t.name)
                xl_img.width = 120
                xl_img.height = 120
                cell = f"{col_letter}{row_idx + 2}"  # +2 = header row + 1-indexing
                ws.add_image(xl_img, cell)
    finally:
        for fn in temp_files:
            try:
                os.remove(fn)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Product Catalog + PG Classifier", layout="wide")
    st.title("Product Catalog + PG Classifier (model only)")

    # Sidebar: model maintenance
    with st.sidebar:
        st.markdown("### PG model")
        if st.button("🔄 Retrain from /training_data"):
            with st.spinner("Retraining – this may take a minute…"):
                result = subprocess.run(
                    [
                        sys.executable, "retrain_pg_model.py",
                        "--input_folder", "training_data",
                        "--output_model", PG_MODEL_PATH,
                    ],
                    capture_output=True, text=True
                )
                st.code(result.stdout + "\n" + result.stderr)
                if result.returncode == 0:
                    load_pg_model.clear()
                    time.sleep(1)
                    st.success("Model retrained and cache cleared. New sessions will use it.")
                else:
                    st.error("Retraining failed – see log above.")

    # Advanced extraction settings
    with st.expander("Advanced PDF extraction"):
        threshold = st.slider("Greyscale threshold", 0, 255, 200, 5)
        min_area_ratio = st.number_input("Min region area ratio", 0.0, 1.0, 0.02, 0.01)
        max_images_per_page = st.number_input("Max images per page", 1, 10, 1, 1)

    pdf_file = st.file_uploader("Upload PDF catalog", type=["pdf"])
    excel_file = st.file_uploader("Upload SKU Excel", type=["xlsx", "xls", "xlsm"])

    # OCR and matching
    use_ocr = st.checkbox(
        "Use OCR to match images to rows (else sequential order)", value=True,
        help="Requires pytesseract + system Tesseract. If unavailable, we fall back to sequential mapping."
    )
    match_col = st.text_input(
        "SKU column to match OCR text against (e.g., 'Product Title (Required)')",
        value="Product Title (Required)"
    )

    if not pdf_file or not excel_file:
        st.info("Upload both a PDF and an Excel file to proceed.")
        return

    # Save uploads
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as fpdf:
        fpdf.write(pdf_file.getvalue())
        pdf_path = fpdf.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as fxls:
        fxls.write(excel_file.getvalue())
        excel_path = fxls.name

    # Extract images
    st.info("Extracting images from PDF…")
    try:
        images = extract_images_from_pdf(
            pdf_path, int(threshold), float(min_area_ratio), int(max_images_per_page)
        )
    except Exception as exc:
        st.error(f"Failed to process PDF: {exc}")
        return
    if not images:
        st.warning("No images extracted with current settings.")
        return
    st.success(f"Extracted {len(images)} image(s).")

    # ── Load SKU data + Predict PGs (MODEL ONLY) ─────────────────────────────
    pg_model = load_pg_model()
    try:
        df_skus = pd.read_excel(excel_path)
        if len(df_skus) == 0:
            st.error("The SKU file is empty.")
            return

        if pg_model is not None:
            # Heuristic: find vendor category + product title columns
            vendor_cat_col = next((c for c in df_skus.columns if "product category" in str(c).lower()), None)
            title_col      = next((c for c in df_skus.columns if "title" in str(c).lower()), None)

            if vendor_cat_col and title_col:
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
                st.info("Could not find Vendor Category and/or Product Title columns – PG prediction skipped.")
                df_skus["PG_pred"] = ""
                df_skus["PG_confidence"] = np.nan
        else:
            st.info("PG model not loaded – predictions unavailable.")
            df_skus["PG_pred"] = ""
            df_skus["PG_confidence"] = np.nan

        # Final value (editable downstream)
        df_skus["PG_final"] = df_skus["PG_pred"]

    except Exception as exc:
        st.error(f"Failed to read Excel file: {exc}")
        return

    # ── Assign images to rows (OCR or sequential) ───────────────────────────
    row_indices = list(range(len(images)))
    if use_ocr and pytesseract is not None and fuzz is not None:
        st.info("Performing OCR and matching to rows…")
        texts = ocr_extract_texts(images)
        if match_col not in df_skus.columns:
            st.warning(f"Match column '{match_col}' not found. Falling back to sequential assignment.")
        else:
            row_indices = match_images_to_skus(texts, df_skus, match_col)

    # ── Show preview (editable PG_pred) ─────────────────────────────────────
    st.subheader("PG suggestions (editable)")
    display_cols = []
    for col in ["PG_pred", "PG_confidence"]:
        if col in df_skus.columns:
            display_cols.append(col)
    # Add helpful context columns if present
    for col in df_skus.columns:
        if any(k in str(col).lower() for k in ["product category", "title", "style id"]):
            display_cols.insert(0, col)
    display_cols = list(dict.fromkeys(display_cols))  # dedupe, keep order

    edited = st.data_editor(
        df_skus[display_cols], num_rows="dynamic", use_container_width=True, key="editor"
    )
    if "PG_pred" in edited.columns:
        df_skus["PG_final"] = edited["PG_pred"]

    # Thumbnails with predicted/edited PG
    st.subheader("Sample extracted images")
    n = min(8, len(images))
    cols = st.columns(n)
    for i in range(n):
        with cols[i]:
            r = row_indices[i] if i < len(row_indices) else i
            label = ""
            if 0 <= r < len(df_skus):
                label = str(df_skus.iloc[r].get("PG_final", df_skus.iloc[r].get("PG_pred", "")))
            st.image(images[i].resize((180, 180)), caption=label if label else f"Image {i+1}")

    # ── Build downloadable Excel with images + PG_final ─────────────────────
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
        st.error(f"Failed to generate Excel: {exc}")
        return
    out.seek(0)
    st.download_button(
        "Download updated Excel",
        data=out.getvalue(),
        file_name="catalog_with_pg.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()

