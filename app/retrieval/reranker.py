
"""
reranker.py
-------------
Retrieval do stages mein hoti hai (ye advanced RAG ka standard pattern hai):

Stage 1 (Retrieve): FAISS/BM25 se jaldi se top ~10-15 "roughly relevant"
chunks nikalo (fast, but thoda approximate)

Stage 2 (Rerank): Ek zyada accurate (but slower) model se un chunks ko
dobara score karo aur sabse best ko upar lao

Cross-encoder kyun better hai plain embeddings se?
Embeddings query aur document ko ALAG-ALAG encode karte hain, phir compare
karte hain (fast but less precise). Cross-encoder query+document ko EK SAATH
dekhta hai aur directly relevance score deta hai (slow but zyada accurate).

Isliye: pehle FAISS se fast filter karo (15 chunks), phir cross-encoder se
un 15 ko precisely rerank karo, top 3-5 hi final answer ke liye use karo.
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict

_reranker_model = None

# ms-marco-MiniLM ek chota, fast cross-encoder hai jo already
# query-document relevance ke liye trained hai
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL)
    return _reranker_model


def rerank(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Chunks ki list leta hai (jo already retrieval se aayi hai) aur
    unhe query ke against dobara, zyada accurately rank karta hai.
    """
    if not chunks:
        return []

    model = get_reranker()

    # Cross-encoder ko [query, document] pairs chahiye
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = model.predict(pairs)

    # Chunks ko naye scores ke saath pair karo aur sort karo
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in scored_chunks[:top_k]:
        result = chunk.copy()
        result["rerank_score"] = float(score)
        results.append(result)

    return results


# --- Quick test ---
if __name__ == "__main__":
    sample_chunks = [
        {"chunk_id": 0, "text": "Machine learning is a subset of AI.", "page": 1},
        {"chunk_id": 1, "text": "I love eating pizza on weekends.", "page": 1},
        {"chunk_id": 2, "text": "Deep learning uses neural networks for image recognition.", "page": 2},
        {"chunk_id": 3, "text": "Neural networks are inspired by the human brain.", "page": 3},
    ]

    query = "How do neural networks work?"
    results = rerank(query, sample_chunks, top_k=2)

    print(f"Query: {query}")
    print("Reranked results:")
    for r in results:
        print(f"  Rerank score: {r['rerank_score']:.4f} | Page: {r['page']} | Text: {r['text']}")