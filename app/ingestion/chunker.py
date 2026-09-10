
"""
chunker.py
-----------
Breaks extracted text into smaller chunks.

Why do we chunk?
- LLMs and embeddings can only handle a limited size of text effectively
- Smaller, focused chunks lead to more accurate retrieval
  (giving a relevant paragraph is better than giving the entire document)

Why do we keep overlap?
- If a sentence gets cut exactly at a chunk boundary, context can be lost
- Keeping a small overlap (e.g., 50 characters) helps reduce this problem
"""

from typing import List, Dict
from app.config import settings


def chunk_text(
    pages_data: List[Dict],
    chunk_size: int = None,
    overlap: int = None
) -> List[Dict]:
    """
    Takes page-wise text and breaks it into chunks, along with page number
    metadata (needed for source citation).

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

            if chunk_str:  # skip empty chunks
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_str,
                    "page": page_num
                })
                chunk_id += 1

            # Start the next chunk with overlap
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