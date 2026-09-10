
"""
keyword_search.py
-------------------
Performs keyword-based search using BM25 — a complement to semantic (FAISS) search.

What is BM25, in simple terms:
It's a statistical formula that checks how often and with how much "importance"
query words appear in a document
(rare words get more weight, common words like "the", "is" get less).

Semantic search captures "meaning", BM25 captures "exact words" —
together they provide better retrieval.
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict


class KeywordSearch:
    def __init__(self, chunks: List[Dict]):
        """
        chunks: list from chunker.py [{"chunk_id":..., "text":..., "page":...}, ...]
        """
        self.chunks = chunks

        # Break each chunk's text into simple words (tokenize)
        # In production you can use a better tokenizer (like nltk),
        # for now simple .split() is sufficient
        tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Takes a query and returns the top_k chunks ranked by BM25 score.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Pair chunks with their scores, sort, and take top_k
        scored_chunks = list(zip(self.chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in scored_chunks[:top_k]:
            if score > 0:  # skip irrelevant zero-score chunks
                result = chunk.copy()
                result["score"] = float(score)
                results.append(result)

        return results


def merge_search_results(
    semantic_results: List[Dict],
    keyword_results: List[Dict],
    semantic_weight: float = 0.6,
    top_k: int = 5
) -> List[Dict]:
    """
    Combines semantic (FAISS) and keyword (BM25) results.

    Why weighted merge? Because semantic search is generally more meaningful,
    so we give it slightly more weight (0.6), and keyword gets 0.4.

    Normalization is necessary because both have scores on different scales
    (semantic: 0-1, BM25: any positive number) — comparing them without
    normalization would be incorrect.
    """
    combined_scores: Dict[int, Dict] = {}

    # Normalize semantic scores (already in 0-1 range, from cosine similarity)
    for r in semantic_results:
        chunk_id = r["chunk_id"]
        combined_scores[chunk_id] = {
            **r,
            "combined_score": r["score"] * semantic_weight
        }

    # Normalize BM25 scores (divide by max score to bring into 0-1 range)
    if keyword_results:
        max_bm25 = max(r["score"] for r in keyword_results) or 1.0
        for r in keyword_results:
            chunk_id = r["chunk_id"]
            normalized_score = (r["score"] / max_bm25) * (1 - semantic_weight)

            if chunk_id in combined_scores:
                combined_scores[chunk_id]["combined_score"] += normalized_score
            else:
                combined_scores[chunk_id] = {
                    **r,
                    "combined_score": normalized_score
                }

    # Sort by combined score
    merged = list(combined_scores.values())
    merged.sort(key=lambda x: x["combined_score"], reverse=True)

    return merged[:top_k]


# --- Quick test ---
if __name__ == "__main__":
    sample_chunks = [
        {"chunk_id": 0, "text": "Machine learning is a subset of AI.", "page": 1},
        {"chunk_id": 1, "text": "I love eating pizza on weekends.", "page": 1},
        {"chunk_id": 2, "text": "Deep learning uses neural networks for pattern recognition.", "page": 2},
    ]

    keyword_search = KeywordSearch(sample_chunks)
    results = keyword_search.search("neural networks", top_k=2)

    print("BM25 keyword search results:")
    for r in results:
        print(f"  Score: {r['score']:.4f} | Page: {r['page']} | Text: {r['text']}")