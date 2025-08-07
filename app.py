"""
Streamlit application for processing product catalog PDFs
-------------------------------------------------------

This application allows a user to upload a PDF product catalog and an Excel
spreadsheet containing SKU information. The app extracts large graphical
regions (assumed to be product photographs) from each page of the PDF,
groups the extracted images into a user‑defined number of clusters using
simple colour histogram features and KMeans clustering, and maps each
extracted image to the corresponding row in the uploaded Excel sheet. The
resulting table, with cluster labels and embedded images, can be downloaded
as a new Excel file.

**Key features**

* **PDF extraction**: Uses `pdf2image` to rasterise pages and `scipy.ndimage`
  to locate large connected regions of non‑white pixels. Only the largest
  regions on each page are assumed to contain product images. The number
  of images to extract per page and the minimum relative area threshold
  can be adjusted through the UI.
* **Feature computation**: Colour histograms are computed for each cropped
  image and normalised. These histograms form the feature vectors used in
  clustering.
* **Clustering**: Unsupervised KMeans clustering (via scikit‑learn) is
  applied to group images into product categories. The number of clusters
  is configurable by the user.
* **Excel integration**: Reads the uploaded SKU spreadsheet with pandas,
  adds a new column for the assigned cluster label, and embeds each
  corresponding product image directly into the sheet using openpyxl.

Limitations
-----------
This example does not rely on OCR or PDF text extraction (since neither
`pytesseract` nor `pdfminer.six` are available in the constrained
environment). Instead, it assumes that the sequence of extracted product
images matches the row order of the provided SKU file. For best results,
ensure that the PDF pages are ordered consistently with the SKU list.
"""

import os
import tempfile
from io import BytesIO
from typing import List, Tuple

import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans

import joblib
import subprocess, sys, time

from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
import openpyxl.utils

# Optional OCR and fuzzy matching libraries. These are imported only if
# available in the runtime environment. To enable OCR‑driven matching
# on Streamlit Cloud you must add `pytesseract` and `rapidfuzz` to your
# requirements.txt and install the Tesseract binary via packages.txt (see
# accompanying documentation for details). If either library is missing,
# the application will fall back to simple row‑order matching.
try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None  # type: ignore
try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:
    fuzz = None  # type: ignore

import openpyxl  # Needed for reading macro‑enabled templates and taxonomy spreadsheets

# `streamlit` is only required when running the interactive application. To
# allow importing this module for testing purposes in environments where
# Streamlit may not be installed, defer its import until within the
# `main()` function.
try:
    import streamlit as st  # type: ignore[assignment]
except ImportError:
    st = None  # type: ignore[assignment]

@st.cache_resource(show_spinner="Loading PG model…")
def load_pg_model(model_path="models/pg_classifier_latest.joblib"):
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def extract_images_from_pdf(
    pdf_path: str,
    threshold: int = 200,
    min_area_ratio: float = 0.02,
    max_images_per_page: int = 1,
) -> List[Image.Image]:
    """Convert the pages of a PDF into images and extract large non‑white regions.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file on disk.
    threshold:
        Greyscale pixel threshold below which a pixel is considered non‑white.
        Higher values are more permissive and will select more of the page.
    min_area_ratio:
        Minimum area (relative to the page) that a region must occupy to be
        considered a valid candidate. Helps filter out small text snippets.
    max_images_per_page:
        Maximum number of candidate regions to extract per page. Regions are
        sorted by area and the largest ones are selected.

    Returns
    -------
    List[Image.Image]
        Cropped PIL images corresponding to detected product images.
    """
    extracted: List[Image.Image] = []
    # Convert each page to an RGB PIL image at a moderate DPI.
    pages = convert_from_path(pdf_path, dpi=200)
    for page in pages:
        # Downsample the page to speed up connected component analysis.
        scale_factor = 4
        small = page.resize((max(1, page.width // scale_factor), max(1, page.height // scale_factor)))
        arr = np.array(small.convert("L"))  # convert to greyscale
        # Create a binary mask where True indicates non‑white pixels.
        binary = arr < threshold
        # Label connected components of the binary image.
        labeled_array, num_features = ndimage.label(binary)
        if num_features == 0:
            continue
        objects = ndimage.find_objects(labeled_array)
        # Compute page area for filtering.
        page_area = small.width * small.height
        # Build list of bounding boxes with their areas.
        bboxes: List[Tuple[int, Tuple[int, int, int, int]]] = []
        for sl in objects:
            # Each slice is a tuple of slice objects for y and x axes.
            y0, y1 = sl[0].start, sl[0].stop
            x0, x1 = sl[1].start, sl[1].stop
            width = x1 - x0
            height = y1 - y0
            area = width * height
            if page_area > 0 and (area / page_area) >= min_area_ratio:
                bboxes.append((area, (x0, y0, x1, y1)))
        if not bboxes:
            continue
        # Sort bounding boxes by descending area and take the largest few.
        bboxes.sort(reverse=True, key=lambda item: item[0])
        selected = bboxes[: max_images_per_page]
        # Extract the selected regions from the full‑resolution page.
        # Compute accurate scaling factors based on the downsampled size.
        if small.width == 0 or small.height == 0:
            continue
        scale_x = page.width / small.width
        scale_y = page.height / small.height
        for _, (x0, y0, x1, y1) in selected:
            # Scale coordinates back to original resolution.
            left = int(x0 * scale_x)
            upper = int(y0 * scale_y)
            right = int(x1 * scale_x)
            lower = int(y1 * scale_y)
            # Clip to page boundaries.
            left = max(left, 0)
            upper = max(upper, 0)
            right = min(right, page.width)
            lower = min(lower, page.height)
            crop = page.crop((left, upper, right, lower))
            extracted.append(crop)
    return extracted


def ocr_extract_texts(images: List[Image.Image]) -> List[str]:
    """Perform OCR on a list of images to extract text.

    If pytesseract is not installed or not available, returns a list of
    empty strings. This function converts each image to greyscale to aid
    OCR and strips whitespace from the resulting text.

    Parameters
    ----------
    images:
        List of PIL images from which to extract text.

    Returns
    -------
    List[str]
        Extracted text for each image (may be empty).
    """
    texts: List[str] = []
    if pytesseract is None:
        # OCR library unavailable; return empty strings for all images.
        return ["" for _ in images]
    for img in images:
        try:
            # Convert to greyscale to improve OCR quality.
            grey = img.convert("L")
            text = pytesseract.image_to_string(grey)
            # Normalise whitespace and convert to lower case for matching.
            text = " ".join(text.strip().split())
            texts.append(text)
        except Exception:
            texts.append("")
    return texts


def match_images_to_skus(
    texts: List[str], df: pd.DataFrame, column: str
) -> List[int]:
    """Match each OCR‑extracted text to a row in the SKU DataFrame.

    For each text string, this function finds the index of the row in
    `df` whose `column` value best matches the text. Matching is
    performed using a fuzzy string similarity metric provided by
    rapidfuzz if available; otherwise a simple case‑insensitive exact
    search is used. Rows are assigned uniquely: once a row is matched,
    it is removed from further consideration. If a text cannot be
    matched (e.g. all remaining rows have very low similarity), it is
    assigned to the next available row index in order.

    Parameters
    ----------
    texts:
        List of strings extracted from product images via OCR.
    df:
        DataFrame containing SKU information.
    column:
        Name of the column in `df` to match against.

    Returns
    -------
    List[int]
        List of row indices (0‑based) in `df` corresponding to each
        input text. Unmatched images receive the first unused index.
    """
    n = len(df)
    available_indices = list(range(n))
    matches: List[int] = []
    # Precompute candidate strings for matching; convert to lower case
    candidates = [str(df.iloc[i][column]) if i < n else "" for i in range(n)]
    candidates_lower = [c.lower() for c in candidates]
    for t in texts:
        if not available_indices:
            matches.append(-1)
            continue
        # Lower‑case OCR text for comparison
        t_lower = t.lower().strip()
        best_idx = available_indices[0]
        best_score = -1.0
        # If rapidfuzz is available, use fuzzy partial_ratio; otherwise use simple containment
        for idx in available_indices:
            c = candidates_lower[idx]
            if fuzz is not None:
                try:
                    score = fuzz.partial_ratio(t_lower, c)
                except Exception:
                    score = 0.0
            else:
                # naive matching: score is 100 if substring found, else 0
                score = 100.0 if t_lower in c else 0.0
            if score > best_score:
                best_score = score
                best_idx = idx
        # Threshold: if the best_score is very low (<30) we consider it a poor match
        # and simply assign the next available row in order. A threshold of 30 is
        # empirically chosen; adjust as needed.
        if fuzz is not None and best_score < 30:
            best_idx = available_indices[0]
        matches.append(best_idx)
        # Remove the matched index
        available_indices.remove(best_idx)
    return matches


def load_taxonomy(taxonomy_file: str | BytesIO) -> pd.DataFrame:
    """Load taxonomy data from an Excel file.

    This function reads a taxonomy Excel file that should contain at least
    the columns 'Industry', 'MD' and 'PG'. If these columns are missing,
    the function will raise a ValueError.

    Parameters
    ----------
    taxonomy_file:
        Path to an Excel file on disk or a BytesIO object containing
        the taxonomy workbook.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['Industry', 'MD', 'MG', 'PG'] or a
        subset thereof.
    """
    try:
        # Use pandas to load the first sheet by default
        df = pd.read_excel(taxonomy_file)
    except Exception as exc:
        raise ValueError(f"Failed to load taxonomy file: {exc}")
    # Normalise column names by stripping whitespace and capitalising
    df.columns = [str(c).strip() for c in df.columns]
    required_cols = {'MD', 'PG'}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Taxonomy must contain at least the columns {required_cols}, but it has {df.columns}."
        )
    return df


def compute_color_histograms(images: List[Image.Image], bins: int = 32) -> np.ndarray:
    """Compute normalised RGB colour histograms for a list of images.

    Each image is resized to a fixed size (128×128) before histogram
    computation. The histograms for the R, G and B channels are concatenated
    to form a feature vector.

    Parameters
    ----------
    images:
        List of PIL images.
    bins:
        Number of histogram bins per channel.

    Returns
    -------
    np.ndarray
        Array of shape (n_images, bins*3) containing the normalised histograms.
    """
    if not images:
        return np.empty((0, bins * 3))
    features: List[np.ndarray] = []
    for img in images:
        # Ensure consistent orientation and size.
        resized = img.resize((128, 128))
        arr = np.array(resized)
        # Compute histograms for each colour channel.
        hist_r, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 255))
        hist_g, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 255))
        hist_b, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 255))
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float64)
        # Normalise to sum to 1 to ensure scale invariance.
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
        features.append(hist)
    return np.vstack(features)


def cluster_images(features: np.ndarray, n_clusters: int) -> np.ndarray:
    """Group images into clusters using KMeans.

    Parameters
    ----------
    features:
        Feature matrix of shape (n_samples, n_features).
    n_clusters:
        Number of clusters to produce.

    Returns
    -------
    np.ndarray
        Array of integer labels assigning each sample to a cluster (0‑based).
    """
    n_samples = features.shape[0]
    if n_samples == 0:
        return np.array([], dtype=int)
    # If there are fewer samples than requested clusters, reduce the number of clusters.
    k = min(n_clusters, n_samples)
    # n_init='auto' uses a sensible default in recent versions of scikit‑learn.
    # In scikit‑learn versions prior to 1.4 the `n_init` parameter must be an
    # integer. Use 10 initialisations by default for robustness.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    return labels


def embed_images_to_excel(
    df: pd.DataFrame,
    images: List[Image.Image],
    cluster_labels: np.ndarray,
    output_stream: BytesIO,
    image_column: str = 'Image',
    category_column: str = 'Category',
    row_indices: List[int] | None = None,
) -> None:
    """Create an Excel file with embedded images and cluster labels.

    This function writes the provided DataFrame to an Excel workbook,
    adds a new column containing categorical labels for each image and
    embeds the images into specific rows determined by `row_indices`.

    Parameters
    ----------
    df:
        DataFrame containing SKU information. It must have at least as many rows
        as there are images if `row_indices` is None; otherwise the largest
        value in `row_indices` must be less than the number of rows.
    images:
        List of PIL images to embed. The number of images should match the
        length of `cluster_labels` and `row_indices` (if provided).
    cluster_labels:
        Array of cluster labels corresponding to each image.
    output_stream:
        BytesIO object to which the Excel file will be written.
    image_column:
        Name of the column where images will be placed. A header with this
        name will be created if it does not exist.
    category_column:
        Name of the column where the cluster labels will be written.
    row_indices:
        Optional list of row indices (0‑based) indicating which rows in the
        DataFrame correspond to each image. If provided, its length must
        equal the number of images. If None, images will be inserted in
        sequential order starting from row 0.
    """
    n_rows = df.shape[0]
    n_images = len(images)
    # Validate sizes
    if row_indices is None:
        if n_rows < n_images:
            raise ValueError(
                f"The SKU file has {n_rows} rows, but {n_images} images were extracted."
            )
        row_indices = list(range(n_images))
    else:
        if len(row_indices) != n_images:
            raise ValueError("Length of row_indices must match number of images.")
        if max(row_indices) >= n_rows:
            raise ValueError("row_indices contains an index outside the DataFrame.")
    # Prepare output DataFrame
    df_out = df.copy().reset_index(drop=True)
    # Initialize category column with empty strings
    if category_column not in df_out.columns:
        df_out[category_column] = [''] * n_rows
    # Determine category labels. If cluster_labels contains numeric labels, produce
    # human‑readable group names; otherwise, use string representation directly.
    categories: List[str] = []
    for label in cluster_labels:
        try:
            categories.append(f"Group {int(label) + 1}")
        except (TypeError, ValueError):
            categories.append(str(label))
    for idx, row_idx in enumerate(row_indices):
        df_out.at[row_idx, category_column] = categories[idx]
    # Ensure image column exists
    if image_column not in df_out.columns:
        df_out[image_column] = [''] * n_rows
    # Write DataFrame to Excel
    temp_files: List[str] = []
    try:
        with pd.ExcelWriter(output_stream, engine='openpyxl') as writer:
            df_out.to_excel(writer, index=False, sheet_name='Sheet1')
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            # Determine column index for images
            image_col_idx = df_out.columns.get_loc(image_column) + 1  # openpyxl uses 1‑based indexing
            col_letter = openpyxl.utils.get_column_letter(image_col_idx)
            # Set column width for images
            worksheet.column_dimensions[col_letter].width = 25
            import tempfile as _tempfile
            for img_idx, img in enumerate(images):
                row_idx = row_indices[img_idx]
                # Save image to a temporary file
                tmp = _tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(tmp, format='PNG')
                tmp.close()
                temp_files.append(tmp.name)
                xl_img = XLImage(tmp.name)
                # Resize image to fit cell
                xl_img.width = 120
                xl_img.height = 120
                # Excel rows are 1‑based and header occupies row 1
                cell_row = row_idx + 2
                cell = f"{col_letter}{cell_row}"
                worksheet.add_image(xl_img, cell)
    finally:
        # Clean up temporary files
        for fn in temp_files:
            try:
                os.remove(fn)
            except OSError:
                pass


def main() -> None:
    st.set_page_config(page_title="Product Catalog Processor", layout="wide")
    st.title("Product Catalog Processor")

# --- Load & manage PG model ---
pg_model = load_pg_model()
with st.sidebar:
    st.markdown('### PG model maintenance')
    if st.button('🔄 Retrain PG model'):
        with st.spinner('Retraining model – please wait...'):
            result = subprocess.run([
                sys.executable, 'retrain_pg_model.py',
                '--input_folder', 'training_data',
                '--output_model', 'models/pg_classifier_latest.joblib'],
                capture_output=True, text=True)
            st.code(result.stdout + '\n' + result.stderr)
            if result.returncode == 0:
                load_pg_model.clear()
                time.sleep(1)
                pg_model = load_pg_model()
                st.success('Model retrained and reloaded for all users.')
            else:
                st.error('Retraining failed – see log above.')

    st.write(
        """
        Upload a product catalog in PDF format along with an Excel file containing
        SKU information. The app will detect product images in the PDF, group
        them into categories via unsupervised clustering and append the
        category labels and the images to your Excel sheet. You can then
        download the enriched workbook.

        **Important**: The number of extracted images should match the number
        of rows in your SKU file. Please ensure that your PDF lists products
        in the same order as the SKUs appear in your spreadsheet.
        """
    )

    pdf_file = st.file_uploader("Upload PDF catalog", type=["pdf"])
    excel_file = st.file_uploader("Upload SKU Excel", type=["xlsx", "xls", "xlsm"])
    taxonomy_file = st.file_uploader(
        "Upload Taxonomy (optional)",
        type=["xlsx", "xls", "xlsm"],
        help="Excel file containing columns 'MD' and 'PG'. If not provided, the app will use a default Taxonomy.xlsx if present."
    )

    with st.expander("Advanced settings"):
        threshold = st.slider(
            "Greyscale threshold (non‑white pixel cutoff)",
            min_value=0,
            max_value=255,
            value=200,
            step=5,
            help="Pixels darker than this greyscale value are considered part of an image."
        )
        min_area_ratio = st.number_input(
            "Minimum relative area for a detected region",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.01,
            help="Regions smaller than this fraction of the page will be ignored."
        )
        max_images_per_page = st.number_input(
            "Maximum number of images to extract per page",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
        )
        n_clusters = st.number_input(
            "Number of product groups (clusters)",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
        )

    if pdf_file is not None and excel_file is not None:
        # Save uploaded files to temporary locations.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(pdf_file.getvalue())
            pdf_path = tmp_pdf.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_excel:
            tmp_excel.write(excel_file.getvalue())
            excel_path = tmp_excel.name

        st.info("Processing the PDF… this may take a moment depending on file size.")
        try:
            images = extract_images_from_pdf(
                pdf_path=pdf_path,
                threshold=int(threshold),
                min_area_ratio=float(min_area_ratio),
                max_images_per_page=int(max_images_per_page),
            )
        except Exception as exc:
            st.error(f"Failed to process PDF: {exc}")
            return
        if not images:
            st.warning("No images were detected in the PDF with the current settings.")
            return
        st.success(f"Extracted {len(images)} candidate image(s) from the PDF.")

        # Compute features and cluster images based on colour
        features = compute_color_histograms(images)
        labels = cluster_images(features, int(n_clusters))
       # --- Load SKU data and run PG prediction safely --------------------------------
try:
    df_skus = pd.read_excel(excel_path)
    if len(df_skus) == 0:
        st.error("The SKU file is empty.")
        return

    # ===== PG predictions (inside the try) =====
    if pg_model is not None:
        # Heuristic column detection
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
        else:
            st.info("Could not find Vendor Category / Product Title columns – PG prediction skipped.")

    # Use editable prediction as final PG for all downstream steps
    if "PG_pred" in df_skus.columns:
        df_skus["PG_final"] = df_skus["PG_pred"]
    else:
        # Fallback to an empty column if prediction didn't run
        df_skus["PG_final"] = ""

except Exception as exc:
    st.error(f"Failed to read Excel file: {exc}")
    return
# -------------------------------------------------------------------------------

      except Exception as exc:
            st.error(f"Failed to read Excel file: {exc}")
            return
        if len(df_skus) == 0:
            st.error("The SKU file is empty.")
            return
        # Determine text columns for matching (object dtype)
        text_columns = [c for c, dt in zip(df_skus.columns, df_skus.dtypes) if dt == object]
        # Load taxonomy data if provided or if a default file exists
        taxonomy_df: pd.DataFrame | None = None
        taxonomy_error = None
        if taxonomy_file is not None:
            try:
                taxonomy_df = load_taxonomy(BytesIO(taxonomy_file.getvalue()))
            except Exception as exc:
                taxonomy_error = str(exc)
        else:
            # Attempt to load a default taxonomy file from working directory
            default_taxonomy_path = os.path.join(os.getcwd(), 'Taxonomy.xlsx')
            if os.path.exists(default_taxonomy_path):
                try:
                    taxonomy_df = load_taxonomy(default_taxonomy_path)
                except Exception as exc:
                    taxonomy_error = str(exc)
        if taxonomy_error:
            st.warning(f"Could not load taxonomy: {taxonomy_error}")
        # Decide whether OCR‑based matching can be used
        ocr_available = (pytesseract is not None and fuzz is not None)
        if ocr_available:
            st.subheader("Matching settings")
            use_ocr = st.checkbox(
                "Use OCR/Text matching to assign images to SKUs", value=True,
                help="If selected, the app will perform OCR on each extracted image and match the resulting text to your chosen column in the SKU sheet."
            )
        else:
            use_ocr = False
        row_indices = list(range(len(images)))
        if use_ocr and ocr_available:
            if not text_columns:
                st.warning(
                    "No textual columns were detected in the SKU file to match against; falling back to sequential assignment."
                )
            else:
                match_col = st.selectbox(
                    "Select the column to match against", options=text_columns,
                    help="Choose the column from your SKU sheet that contains product names or descriptions."
                )
                # Run OCR on each image
                st.info("Performing OCR on extracted images…")
                texts = ocr_extract_texts(images)
                # Compute matching indices based on fuzzy similarity
                row_indices = match_images_to_skus(texts, df_skus, match_col)
        # Select MDs (Merchandise Divisions) if taxonomy is available
        selected_mds: List[str] = []
        pg_choices: List[str] = []
        if taxonomy_df is not None and not taxonomy_df.empty:
            md_list = sorted(set(taxonomy_df['MD'].dropna().astype(str)))
            st.subheader("Select Merchandise Divisions (MDs)")
            selected_mds = st.multiselect(
                "Choose which MDs to include when selecting PG categories.",
                options=md_list,
                default=md_list,
            )
            # Filter taxonomy to selected MDs and get list of PGs
            if selected_mds:
                taxonomy_filtered = taxonomy_df[taxonomy_df['MD'].astype(str).isin(selected_mds)]
            else:
                taxonomy_filtered = taxonomy_df
            pg_choices = sorted(set(taxonomy_filtered['PG'].dropna().astype(str)))
        # If no taxonomy is loaded or no PG choices, fall back to generic group names
        cluster_to_pg: dict[int, str] = {}
        if pg_choices:
            st.subheader("Assign Product Groups (PGs) to clusters")
            # Identify vendor product category and product title columns
            vendor_cat_col = None
            title_col = None
            for col in df_skus.columns:
                col_lower = col.lower()
                if vendor_cat_col is None and 'product category' in col_lower:
                    vendor_cat_col = col
                if title_col is None and 'title' in col_lower:
                    title_col = col
            # Precompute OCR texts for each image (empty if OCR unavailable)
            try:
                ocr_texts = ocr_extract_texts(images)
            except Exception:
                ocr_texts = ["" for _ in images]
            # Build lowercase PG list for matching
            pg_choices_lower = [pg.lower() for pg in pg_choices]
            suggestions: dict[int, str] = {}
            if fuzz is not None:
                # Compute suggestion per cluster by aggregating row‑level scores
                # Weights for vendor category, product title and OCR text
                w_cat, w_title, w_ocr = 0.6, 0.3, 0.1
                for cluster_id in sorted(set(labels)):
                    # Collect indices of images and corresponding row indices in this cluster
                    indices_in_cluster = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                    # Initialise score accumulator for each PG
                    pg_scores = {pg: 0.0 for pg in pg_choices}
                    for idx in indices_in_cluster:
                        row_idx = row_indices[idx]
                        # Vendor category text
                        cat_text = ''
                        if vendor_cat_col is not None and 0 <= row_idx < len(df_skus):
                            val = df_skus.iloc[row_idx][vendor_cat_col]
                            if pd.notna(val):
                                cat_text = str(val).lower()
                        # Product title text
                        title_text = ''
                        if title_col is not None and 0 <= row_idx < len(df_skus):
                            val = df_skus.iloc[row_idx][title_col]
                            if pd.notna(val):
                                title_text = str(val).lower()
                        # OCR text
                        ocr_text = ocr_texts[idx].lower() if idx < len(ocr_texts) else ''
                        for pg_name, pg_lower in zip(pg_choices, pg_choices_lower):
                            score = 0.0
                            try:
                                if cat_text:
                                    score += w_cat * fuzz.partial_ratio(cat_text, pg_lower)
                                if title_text:
                                    score += w_title * fuzz.partial_ratio(title_text, pg_lower)
                                if ocr_text:
                                    score += w_ocr * fuzz.partial_ratio(ocr_text, pg_lower)
                            except Exception:
                                score += 0.0
                            pg_scores[pg_name] += score
                    # Choose the PG with the highest aggregated score
                    best_pg = max(pg_scores.items(), key=lambda x: x[1])[0] if pg_scores else None
                    if best_pg is not None:
                        suggestions[cluster_id] = best_pg
            # For each cluster, prompt user to select PG with default suggestion if available
            unique_clusters = sorted(set(labels))
            for cluster_id in unique_clusters:
                default_idx = 0
                if cluster_id in suggestions:
                    try:
                        default_idx = pg_choices.index(suggestions[cluster_id])
                    except ValueError:
                        default_idx = 0
                cluster_to_pg[cluster_id] = st.selectbox(
                    f"Cluster {cluster_id + 1} → PG",
                    options=pg_choices,
                    index=default_idx,
                    key=f"pg_select_{cluster_id}"
                )
        else:
            # No PG choices available; use generic group naming
            cluster_to_pg = {c: f"Group {int(c) + 1}" for c in sorted(set(labels))}
        # Provide a preview of assignments
        preview_df = df_skus.copy().reset_index(drop=True)
        # Create or reset the PG column
        pg_column_name = 'PG'
        preview_df[pg_column_name] = [''] * len(preview_df)
        for i, row_idx in enumerate(row_indices):
            if 0 <= row_idx < len(preview_df):
                # Determine PG assignment for this image based on cluster mapping
                cluster_id = labels[i]
                pg_value = cluster_to_pg.get(cluster_id, f"Group {int(cluster_id) + 1}")
                preview_df.at[row_idx, pg_column_name] = pg_value
        st.subheader("Preview of SKU assignments (with PG)")
        st.subheader('PG suggestions (editable)')
editable_cols = list(preview_df.columns)
if 'PG_pred' in df_skus.columns and 'PG_pred' not in editable_cols:
    editable_cols.append('PG_pred')
if 'PG_confidence' in df_skus.columns and 'PG_confidence' not in editable_cols:
    editable_cols.append('PG_confidence')
preview_df = st.data_editor(preview_df[editable_cols], num_rows='dynamic', use_container_width=True)
df_skus['PG_final'] = preview_df.get('PG_pred', df_skus.get('PG_pred', ''))

        # Show a few thumbnail images with their cluster labels (display PG names)
        st.subheader("Sample extracted images")
        num_display = min(8, len(images))
        thumbs = [img.resize((150, 150)) for img in images[:num_display]]
        captions = []
for i in range(num_display):
    r = row_indices[i] if i < len(row_indices) else i
    val = ''
    if 0 <= r < len(df_skus):
        val = str(df_skus.iloc[r].get('PG_final', df_skus.iloc[r].get('PG_pred', '')))
    captions.append(val if val else f'Image {i+1}')
        cols = st.columns(num_display)
        for idx in range(num_display):
            with cols[idx]:
                st.image(thumbs[idx], caption=captions[idx])
        # Prepare downloadable Excel
        output_stream = BytesIO()
        try:
            # Convert cluster labels into PG labels for embedding
            pg_labels = np.array([df_skus.iloc[idx].get('PG_final', df_skus.iloc[idx].get('PG_pred', '')) if 0 <= idx < len(df_skus) else '' for idx in row_indices])
            embed_images_to_excel(
                df=df_skus,
                images=images,
                cluster_labels=pg_labels,
                output_stream=output_stream,
                image_column='Image',
                category_column='PG_final',
                row_indices=row_indices,
            )
        except Exception as exc:
            st.error(f"Failed to generate output Excel: {exc}")
            return
        output_stream.seek(0)
        st.download_button(
            label="Download updated Excel",
            data=output_stream.getvalue(),
            file_name="catalog_with_images.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
