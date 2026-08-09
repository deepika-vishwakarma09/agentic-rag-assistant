"""
vector_store.py
-----------------
FAISS ka kaam hai embeddings ko store karna aur "sabse similar chunks
ढूंढो" wala fast search dena.

FAISS kyun? Kyunki jab hazaaron chunks ho jayein, to har ek se
manually compare karna slow ho jaata hai. FAISS ek "index" banata hai
jisse search bahut fast ho jaati hai.
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from app.config import settings


class VectorStore:
    def __init__(self, dimension: int = 384):
        # 384 = all-MiniLM-L6-v2 model ka output size
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (cosine, kyunki normalized hai)
        self.chunks_metadata: List[Dict] = []  # chunk_id -> {text, page} mapping

    def add_chunks(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Embeddings aur unke metadata (text, page number) ko store karta hai.
        """
        self.index.add(embeddings)
        self.chunks_metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Query embedding ke sabse similar top_k chunks dhundta hai.

        Returns: [{"text": "...", "page": 1, "score": 0.85}, ...]
        Score jitna 1 ke paas, utna zyada relevant.
        """
        if self.index.ntotal == 0:
            return []

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS -1 deta hai agar enough results na ho
                continue
            chunk_data = self.chunks_metadata[idx].copy()
            chunk_data["score"] = float(score)
            results.append(chunk_data)

        return results

    def save(self, path: str = None):
        """Index aur metadata disk pe save karo (baar baar re-embed na karna pade)."""
        path = path or settings.VECTOR_DB_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_metadata.pkl", "wb") as f:
            pickle.dump(self.chunks_metadata, f)

    def load(self, path: str = None):
        """Pehle se saved index load karo."""
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