"""
vector_store.py
-----------------
FAISS is responsible for storing embeddings and providing fast
"find the most similar chunks" search.

Why FAISS? Because when there are thousands of chunks, manually comparing
each one becomes slow. FAISS creates an "index" that makes
search very fast.
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from app.config import settings


class VectorStore:
    def __init__(self, dimension: int = 384):
        # 384 = output size of the all-MiniLM-L6-v2 model
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (cosine, since vectors are normalized)
        self.chunks_metadata: List[Dict] = []  # chunk_id -> {text, page} mapping

    def add_chunks(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Stores the embeddings and their metadata (text, page number).
        """
        self.index.add(embeddings)
        self.chunks_metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Finds the top_k chunks most similar to the query embedding.

        Returns: [{"text": "...", "page": 1, "score": 0.85}, ...]
        The closer the score is to 1, the more relevant.
        """
        if self.index.ntotal == 0:
            return []

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 if there aren't enough results
                continue
            chunk_data = self.chunks_metadata[idx].copy()
            chunk_data["score"] = float(score)
            results.append(chunk_data)

        return results

    def save(self, path: str = None):
        """Save the index and metadata to disk (so re-embedding isn't needed every time)."""
        path = path or settings.VECTOR_DB_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_metadata.pkl", "wb") as f:
            pickle.dump(self.chunks_metadata, f)

    def load(self, path: str = None):
        """Load a previously saved index."""
        path = path or settings.VECTOR_DB_PATH

        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_metadata.pkl", "rb") as f:
            self.chunks_metadata = pickle.load(f)


# --- Quick test ---
if __name__ == "__main__":
    from app.ingestion.embedder import embed_chunks, embed_query

    sample_chunks = [
        {"chunk_id": 0, "text": "Machine learning is a subset of AI.", "page": 1},
        {"chunk_id": 1, "text": "I love eating pizza on weekends.", "page": 1},
        {"chunk_id": 2, "text": "Deep learning uses neural networks.", "page": 2},
    ]

    embeddings = embed_chunks(sample_chunks)

    store = VectorStore(dimension=embeddings.shape[1])
    store.add_chunks(embeddings, sample_chunks)

    query_emb = embed_query("Tell me about neural networks")
    results = store.search(query_emb, top_k=2)

    print("Top matches:")
    for r in results:
        print(f"  Score: {r['score']:.4f} | Page: {r['page']} | Text: {r['text']}") 