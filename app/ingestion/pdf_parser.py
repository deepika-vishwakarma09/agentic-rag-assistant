
"""
pdf_parser.py
--------------
PDF se text extract karne ka kaam yahan hota hai.
Har page ka text nikaal ke, page number ke saath return karte hain
taaki baad mein source citation ("page 3 se aaya") dikha saken.
"""

import fitz  # PyMuPDF
from typing import List, Dict


def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """
    PDF file path leta hai, aur har page ka text nikaal ke
    list of dicts return karta hai: [{"page": 1, "text": "..."}, ...]
    """
    pages_data = []

    try:
        doc = fitz.open(file_path)

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()

            # Empty pages (jaise sirf image ho) skip kar do abhi ke liye
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
                          "Ho sakta hai ye scanned PDF ho — OCR chahiye hoga.")

    return pages_data


def get_full_text(pages_data: List[Dict]) -> str:
    """
    Sare pages ka text combine karke ek single string return karta hai.
    Chunking se pehle isko use karenge.
    """
    return "\n\n".join([p["text"] for p in pages_data])


# --- Quick test (isko directly run karke check kar sakti ho) ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    result = extract_text_from_pdf(sys.argv[1])
    print(f"Total pages with text: {len(result)}")
    print(f"First page preview:\n{result[0]['text'][:300]}")