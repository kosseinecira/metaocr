"""
PDF Title Extraction Pipeline
-----------------------------
This script automatically extracts document titles from PDF files located in the ./books directory.
It first tries to use PDF metadata. If unavailable, it applies OCR-based title detection using PaddleOCR.

Main steps:
1. Read PDFs from ./books
2. Extract title metadata (if available)
3. Capture the first page of the PDF as an image
4. Run document layout detection to find the title coordinates
5. Crop the title area and save it as an image
6. Perform OCR on the cropped title image to recognize the title text
7. Print the recognized title
"""

import re
from datetime import datetime
import pdfplumber
from pathlib import Path
from paddleocr import LayoutDetection, PaddleOCR, TextRecognition
from PIL import Image

# Define input and output directories
folder_path = Path("./books")
output_path = Path("./output")
print(folder_path)

# ----------------------------------------------
#  Utility: Format metadata-based title strings
# ----------------------------------------------
def form_title(title, author, creation_date):
    """
    Construct a clean title string from metadata fields.
    Example: "Deep_Learning_[Ian_Goodfellow_2016-09-01_00-00-00]"
    """
    # Extract and format date if it matches pattern like D:20230901000000Z
    match = re.search(r'D:(\d{14})', creation_date)
    if match:
        dt = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
        creation_date = dt.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        creation_date = re.sub(r'[^\w]+', '_', creation_date.strip())

    # Clean up title and author (replace spaces and special chars)
    title = re.sub(r'[^\w]+', '_', title.strip())
    author = re.sub(r'[^\w]+', '_', author.strip())

    # Assemble formatted name
    formatted = f"{title}[{author}_{creation_date}]"
    formatted = re.sub(r'_+', '_', formatted).strip('_')
    return formatted


# ----------------------------------------------
#  Utility: Get list of all PDF files in folder
# ----------------------------------------------
def loop_dir(folder):
    pdf_files = [f for f in folder.glob("*.pdf") if f.is_file()]
    if not pdf_files:
        print("No PDF found in the path you provided.")
    return pdf_files


pdf_files_paths = loop_dir(folder_path)
print(pdf_files_paths)


# ----------------------------------------------
#  Step 1: Detect title coordinates and crop it
# ----------------------------------------------
def detect_title_and_crop(safe_name, image_dir, first_page_capture_path):
    """
    Runs PaddleOCR LayoutDetection model on the captured first page
    and crops out the detected title region based on coordinates.
    """
    title_image_path = image_dir / "title.jpg"

    # Run layout detection
    model = LayoutDetection(model_name="PP-DocLayout_plus-L")
    output = model.predict(str(first_page_capture_path), batch_size=1, layout_nms=True)

    if not output:
        print(f"⚠️ LayoutDetection returned empty output for {first_page_capture_path}")
        return None

    response = output[0]

    # Search for box labeled as 'doc_title'
    title_box = None
    for box in response['boxes']:
        if box['label'] == 'doc_title':
            title_box = box['coordinate']
            image = Image.open(first_page_capture_path).convert("RGB")
            x1, y1, x2, y2 = map(int, title_box)

            # Crop and save title image
            cropped = image.crop((x1, y1, x2, y2))
            cropped.save(title_image_path)
            print(f"✅ Title image of {safe_name} detected, cropped, and saved as {title_image_path}")
            break

    if not title_box:
        print("❌ No title detected! Bypassing current PDF.")
        return None

    return title_image_path


# ----------------------------------------------
#  Step 2: Recognize title text via OCR
# ----------------------------------------------
def title_recognition(title_image_path):
    """
    Uses PaddleOCR to extract text from the cropped title image.
    Prints the full recognized title and confidence scores.
    """
    ocr = PaddleOCR(lang='en')
    result = ocr.predict(str(title_image_path))

    if not result:
        print("⚠️ No OCR result returned.")
        return

    res = result[0]
    print("Recognized text:", res['rec_texts'])
    print("Confidence scores:", res['rec_scores'])

    full_title = "_".join(res['rec_texts'])
    print("Full title:", full_title)


# ----------------------------------------------
#  Step 3: Capture first page as image
# ----------------------------------------------
def capture_title(page, pdf_current_name):
    """
    Captures the first page of a PDF and saves it as an image.
    Then triggers layout detection and OCR recognition steps.
    """
    safe_name = re.sub(r'[^\w.-]', '_', pdf_current_name.replace('.pdf', ''))

    # Create output folder for this PDF
    image_dir = Path(f"./output/{safe_name}")
    image_dir.mkdir(parents=True, exist_ok=True)

    # Save first page as an image
    first_page_capture_path = image_dir / "first_page.jpg"
    page.to_image(resolution=200).save(first_page_capture_path)
    print(f"✅ First page of {pdf_current_name} captured and saved as {first_page_capture_path}")

    # Detect title and run OCR
    title_image_path = detect_title_and_crop(safe_name, image_dir, first_page_capture_path)
    if title_image_path:
        title_recognition(title_image_path)


# ----------------------------------------------
#  Step 4: Main PDF processing logic
# ----------------------------------------------
def processPdf(pdf_path):
    """
    Process a single PDF:
      1. Try extracting title from metadata.
      2. If not available, analyze first page.
      3. If scanned (image-based), apply OCR detection.
    """
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) == 0:
            return "No pages found in this PDF"

        # --- Case 1: Try metadata ---
        metadata_title = pdf.metadata.get("Title", "")
        metadata_author = pdf.metadata.get("Author", "")
        metadata_creation_date = pdf.metadata.get("CreationDate", "")

        if metadata_title:
            return form_title(metadata_title, metadata_author, metadata_creation_date)

        # --- Case 2: Use first page ---
        first_page = pdf.pages[0]
        has_text = bool(first_page.extract_text())
        has_image = len(first_page.images) > 0

        if not has_text and not has_image:
            return "Page 1 is empty"

        # --- Case 3: If it's scanned (image-based) ---
        if has_image:
            print(f"Page contains image(s) — OCR required")
            capture_title(first_page, pdf_path.name)

        # --- Fallback ---
        return "System unable to read the first page"


# ----------------------------------------------
#  Step 5: Loop over all PDFs in folder
# ----------------------------------------------
for pdf_path in pdf_files_paths:
    print("****************************** PDF:", pdf_path.name, "******************************")
    print(processPdf(pdf_path))
