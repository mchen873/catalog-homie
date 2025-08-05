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

from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
import openpyxl.utils

# `streamlit` is only required when running the interactive application. To
# allow importing this module for testing purposes in environments where
# Streamlit may not be installed, defer its import until within the
# `main()` function.
try:
    import streamlit as st  # type: ignore[assignment]
except ImportError:
    st = None  # type: ignore[assignment]


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
) -> None:
    """Create an Excel file with embedded images and cluster labels.

    Parameters
    ----------
    df:
        DataFrame containing SKU information. It must have at least as many rows
        as there are images.
    images:
        List of PIL images to embed. The number of images should match the
        length of `cluster_labels`.
    cluster_labels:
        Array of cluster labels corresponding to each image.
    output_stream:
        BytesIO object to which the Excel file will be written.
    image_column:
        Name of the column where images will be placed. A header with this
        name will be created if it does not exist.
    category_column:
        Name of the column where the cluster labels will be written.
    """
    n_rows = df.shape[0]
    n_images = len(images)
    if n_rows < n_images:
        raise ValueError(
            f"The SKU file has {n_rows} rows, but {n_images} images were extracted."
        )
    # Prepare output DataFrame
    df_out = df.copy().reset_index(drop=True)
    # Add category labels (1‑based indexing for human readability)
    categories = [f"Group {int(label) + 1}" for label in cluster_labels]
    df_out[category_column] = categories + [''] * (n_rows - n_images)
    # Ensure image column exists
    if image_column not in df_out.columns:
        df_out[image_column] = ''
    # Write DataFrame to Excel
    # Use a list to record temporary image files so they can be cleaned up
    temp_files: List[str] = []
    try:
        with pd.ExcelWriter(output_stream, engine='openpyxl') as writer:
            df_out.to_excel(writer, index=False, sheet_name='Sheet1')
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            # Determine column index for images
            image_col_idx = df_out.columns.get_loc(image_column) + 1  # openpyxl uses 1‑based indexing
            # Set a wider column width to accommodate images
            col_letter = openpyxl.utils.get_column_letter(image_col_idx)
            worksheet.column_dimensions[col_letter].width = 25
            # Insert each image into the corresponding row
            import tempfile as _tempfile
            for idx, img in enumerate(images):
                # Save each image to a temporary file because openpyxl needs a file path
                tmp = _tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(tmp, format='PNG')
                tmp.close()
                temp_files.append(tmp.name)
                xl_img = XLImage(tmp.name)
                # Resize the image to fit within the cell boundaries
                xl_img.width = 120
                xl_img.height = 120
                cell = f"{col_letter}{idx + 2}"  # header occupies row 1
                worksheet.add_image(xl_img, cell)
            # End of with-block triggers writer.save and writer.close
    finally:
        # Remove temporary files after the workbook has been saved
        for fn in temp_files:
            try:
                os.remove(fn)
            except OSError:
                pass


def main() -> None:
    st.set_page_config(page_title="Product Catalog Processor", layout="wide")
    st.title("Product Catalog Processor")
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
    excel_file = st.file_uploader("Upload SKU Excel", type=["xlsx", "xls"])

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

        # Compute features and cluster
        features = compute_color_histograms(images)
        labels = cluster_images(features, int(n_clusters))
        # Load SKU data
        try:
            df_skus = pd.read_excel(excel_path)
        except Exception as exc:
            st.error(f"Failed to read Excel file: {exc}")
            return
        if len(df_skus) < len(images):
            st.warning(
                f"There are {len(images)} extracted images but only {len(df_skus)} rows in the SKU file."
            )
        # Present preview of assignments
        preview_df = df_skus.copy().reset_index(drop=True)
        preview_df['Category'] = [f"Group {l + 1}" for l in labels] + [''] * (len(preview_df) - len(labels))
        st.subheader("Preview of SKU assignments")
        st.dataframe(preview_df)
        # Show a few thumbnail images with their cluster labels
        st.subheader("Sample extracted images")
        num_display = min(8, len(images))
        thumbs = [img.resize((150, 150)) for img in images[:num_display]]
        captions = [f"Group {labels[i] + 1}" for i in range(num_display)]
        cols = st.columns(num_display)
        for idx in range(num_display):
            with cols[idx]:
                st.image(thumbs[idx], caption=captions[idx])
        # Prepare downloadable Excel
        output_stream = BytesIO()
        try:
            embed_images_to_excel(
                df=df_skus,
                images=images,
                cluster_labels=labels,
                output_stream=output_stream,
                image_column='Image',
                category_column='Category',
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