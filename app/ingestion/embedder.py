
"""
embedder.py
------------
Text chunks ko embeddings (numbers ki list) mein convert karta hai.

Embedding kya hai, simple example se samjho:
"cat" aur "kitten" ke embeddings ek dusre ke paas honge (similar meaning)
"cat" aur "car" ke embeddings door honge (different meaning)
Isi "closeness" ka use search ke liye karte hain.

Hum "all-MiniLM-L6-v2" use kar rahe hain kyunki:
- Free hai, local pe chalta hai (koi API cost nahi)
- Fast hai (384-dimension vectors, chota size)
- Achi accuracy deta hai general text ke liye
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
from app.config import settings

# Model ek baar load hota hai (global), baar baar load na ho isliye
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_chunks(chunks: List[Dict]) -> np.ndarray:
    """
    Chunks ki list leta hai (jo chunker.py se aayi thi) aur
    unka embedding matrix return karta hai.

    Input: [{"chunk_id": 0, "text": "...", "page": 1}, ...]
    Output: numpy array of shape (num_chunks, 384)
    """
    model = get_model()
    texts = [chunk["text"] for chunk in chunks]

    # convert_to_numpy=True taaki FAISS mein directly daal saken
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True  # cosine similarity ke liye zaroori
    )

    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    User ka question aane par usko bhi embedding mein convert karna hota hai
    taaki uska matching chunks ke saath comparison ho sake.
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

    # Similarity check (dono normalize hain to bas dot product = cosine similarity)
    similarity_with_ml = np.dot(embeddings[0], query_emb[0])
    similarity_with_pizza = np.dot(embeddings[1], query_emb[0])
    print(f"Similarity with ML sentence: {similarity_with_ml:.4f}")
    print(f"Similarity with pizza sentence: {similarity_with_pizza:.4f}")
    print("(ML wala score zyada hona chahiye, kyunki query bhi AI ke baare mein hai)")