# 📘 PDF Title Extraction and Recognition Pipeline

This project automates **title extraction** from PDF documents using **PaddleOCR** and **LayoutDetection** models.  
It supports both digital and scanned PDFs by detecting title regions, cropping them, and performing OCR recognition to extract the title text.

---

## 🧠 Overview

This script performs a sequence of intelligent operations to extract book or document titles from PDF files:

1. **Scan the folder** for `.pdf` files.  
2. **Open each PDF** and check metadata (Title, Author, CreationDate).  
3. **If metadata exists**, format and return it.  
4. **If not**, process the first page:
   - Capture the first page as an image.
   - Run **layout detection** to find the title block.
   - Crop the detected title area.
   - Run **OCR** to recognize the title text.
5. Save all outputs (first page + cropped title image) in the `./output/` directory.

---

## 🏗️ Project Structure

├── books/ # Folder containing input PDF files
├── output/ # Generated images and results
├── loop_dir.py # Main Python script (algorithm)
├── requirements.txt # Dependencies
└── README.md # This file

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/kosseinecira/metaocr.git
cd metaocr
### 1. Run the script inside a container
docker run --name paddleocr -v "%cd%:/paddle" --shm-size=8G -it paddlepaddle/paddle:3.0.0 /bin/bash
docker start -ai paddleocr
cd /paddle
pip install "paddlex[ocr]"
pip install -U "paddlepaddle>=2.6.0" "paddlex[ocr]>=3.0.0" paddleocr==3.3.0
pip install pdfplumber
