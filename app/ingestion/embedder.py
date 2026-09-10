
"""
embedder.py
------------
Converts text chunks into embeddings (list of numbers).

What is an embedding? A simple example:
The embeddings for "cat" and "kitten" will be close to each other (similar meaning)
The embeddings for "cat" and "car" will be far apart (different meaning)
We use this "closeness" for search.

We are using "all-MiniLM-L6-v2" because:
- It's free and runs locally (no API cost)
- It's fast (384-dimension vectors, small size)
- It gives good accuracy for general text
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
from app.config import settings

# Model is loaded once (global) to avoid loading it repeatedly
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_chunks(chunks: List[Dict]) -> np.ndarray:
    """
    Takes a list of chunks (from chunker.py) and returns
    their embedding matrix.

    Input: [{"chunk_id": 0, "text": "...", "page": 1}, ...]
    Output: numpy array of shape (num_chunks, 384)
    """
    model = get_model()
    texts = [chunk["text"] for chunk in chunks]

    # convert_to_numpy=True so we can directly insert into FAISS
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True  # required for cosine similarity
    )

    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    When a user's question arrives, it also needs to be converted into
    an embedding so it can be compared with the stored chunks.
    """
    model = get_model()
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embedding


# --- Quick test ---
if __name__ == "__main__":
    sample_chunks = [
        {"chunk_id": 0, "text": "Machine learning is a subset of AI.", "page": 1},
        {"chunk_id": 1, "text": "I love eating pizza on weekends.", "page": 1},
    ]
    embeddings = embed_chunks(sample_chunks)
    print(f"Embeddings shape: {embeddings.shape}")  # should be (2, 384)

    query_emb = embed_query("What is AI?")
    print(f"Query embedding shape: {query_emb.shape}")

    # Similarity check (both are normalized, so dot product = cosine similarity)
    similarity_with_ml = np.dot(embeddings[0], query_emb[0])
    similarity_with_pizza = np.dot(embeddings[1], query_emb[0])
    print(f"Similarity with ML sentence: {similarity_with_ml:.4f}")
    print(f"Similarity with pizza sentence: {similarity_with_pizza:.4f}")
    print("(ML score should be higher, because the query is also about AI)")