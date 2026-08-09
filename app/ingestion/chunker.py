
"""
chunker.py
-----------
Extracted text ko chhote chunks mein todne ka kaam.

Kyun chunk karte hain?
- LLM aur embeddings ek limited size ka text hi acche se handle karte hain
- Chote, focused chunks se retrieval zyada accurate hoti hai
  (pura document dene se better ek relevant paragraph dena)

Overlap kyun rakhte hain?
- Agar ek sentence exactly chunk boundary pe kat jaye, to context toot sakta hai
- Thoda overlap (jaise 50 characters) rakhne se ye problem kam ho jaati hai
"""

from typing import List, Dict
from app.config import settings


def chunk_text(
    pages_data: List[Dict],
    chunk_size: int = None,
    overlap: int = None
) -> List[Dict]:
    """
    Page-wise text leta hai aur chunks mein todta hai, page number
    metadata ke saath (source citation ke liye zaroori).

    Returns: [{"chunk_id": 0, "text": "...", "page": 1}, ...]
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    all_chunks = []
    chunk_id = 0

    for page_data in pages_data:
        page_num = page_data["page"]
        text = page_data["text"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_str = text[start:end].strip()

            if chunk_str:  # empty chunk skip karo
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_str,
                    "page": page_num
                })
                chunk_id += 1

            # Overlap ke saath next chunk shuru karo
            start += (chunk_size - overlap)

    return all_chunks


# --- Quick test ---
if __name__ == "__main__":
    sample_pages = [
        {"page": 1, "text": "A" * 1200},  # dummy long text
    ]
    chunks = chunk_text(sample_pages)
    print(f"Total chunks created: {len(chunks)}")
    for c in chunks[:3]:
        print(f"Chunk {c['chunk_id']} (page {c['page']}): length={len(c['text'])}")