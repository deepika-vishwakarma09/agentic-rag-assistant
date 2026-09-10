
"""
pdf_parser.py
--------------
Handles text extraction from PDFs.
Extracts text from each page and returns it along with the page number
so that source citation ("came from page 3") can be shown later.
"""

import fitz  # PyMuPDF
from typing import List, Dict


def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """
    Takes a PDF file path and extracts text from each page,
    returning a list of dicts: [{"page": 1, "text": "..."}, ...]
    """
    pages_data = []

    try:
        doc = fitz.open(file_path)

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()

            # Skip empty pages (e.g., pages with only images) for now
            if text:
                pages_data.append({
                    "page": page_num,
                    "text": text
                })

        doc.close()

    except Exception as e:
        raise ValueError(f"PDF parsing failed: {str(e)}")

    if not pages_data:
        raise ValueError("No extractable text found in PDF. "
                          "This might be a scanned PDF — OCR may be needed.")

    return pages_data


def get_full_text(pages_data: List[Dict]) -> str:
    """
    Combines text from all pages and returns a single string.
    This will be used before chunking.
    """
    return "\n\n".join([p["text"] for p in pages_data])


# --- Quick test (you can run this directly to check) ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    result = extract_text_from_pdf(sys.argv[1])
    print(f"Total pages with text: {len(result)}")
    print(f"First page preview:\n{result[0]['text'][:300]}")