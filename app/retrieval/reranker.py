
"""
reranker.py
-------------
Retrieval happens in two stages (this is a standard advanced RAG pattern):

Stage 1 (Retrieve): Quickly extract the top ~10-15 "roughly relevant"
chunks from FAISS/BM25 (fast, but somewhat approximate)

Stage 2 (Rerank): Re-score those chunks with a more accurate (but slower)
model and bring the best ones to the top

Why is a cross-encoder better than plain embeddings?
Embeddings encode the query and document SEPARATELY, then compare
(fast but less precise). A cross-encoder looks at the query+document
TOGETHER and directly gives a relevance score (slow but more accurate).

Therefore: first filter quickly with FAISS (15 chunks), then precisely
rerank those 15 with the cross-encoder, and use only the top 3-5 for the final answer.
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict

_reranker_model = None

# ms-marco-MiniLM is a small, fast cross-encoder that is already
# trained for query-document relevance
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL)
    return _reranker_model


def rerank(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Takes a list of chunks (already from retrieval) and
    re-ranks them more accurately against the query.
    """
    if not chunks:
        return []

    model = get_reranker()

    # Cross-encoder needs [query, document] pairs
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = model.predict(pairs)

    # Pair chunks with new scores and sort
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