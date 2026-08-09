
"""
keyword_search.py
-------------------
BM25 se keyword-based search karta hai — semantic (FAISS) search ka
complement hai.

BM25 kya hai, simple mein:
Ye ek statistical formula hai jo dekhta hai ki query ke words kitni
baar aur kitni "importance" ke saath document mein aaye hain
(rare words ko zyada weight milta hai, common words jaise "the", "is" ko kam).

Semantic search "meaning" pakadta hai, BM25 "exact words" pakadta hai —
dono milke better retrieval dete hain.
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict


class KeywordSearch:
    def __init__(self, chunks: List[Dict]):
        """
        chunks: chunker.py se aayi list [{"chunk_id":..., "text":..., "page":...}, ...]
        """
        self.chunks = chunks

        # Har chunk ke text ko simple words mein todo (tokenize)
        # Production mein better tokenizer use kar sakte ho (jaise nltk),
        # abhi ke liye simple .split() kaafi hai
        tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query leta hai, BM25 score ke hisaab se top_k chunks return karta hai.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Scores ke saath chunks ko pair karo, sort karo, top_k lo
        scored_chunks = list(zip(self.chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in scored_chunks[:top_k]:
            if score > 0:  # zero-score wale irrelevant chunks skip karo
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
    Semantic (FAISS) aur keyword (BM25) results ko combine karta hai.

    Kyun weighted merge? Kyunki semantic search generally zyada meaningful
    hota hai, isliye usko thoda zyada weight (0.6) dete hain, keyword ko 0.4.

    Normalization zaroori hai kyunki dono ke scores alag scale mein hote hain
    (semantic: 0-1, BM25: koi bhi positive number) — bina normalize kiye
    compare karna galat hoga.
    """
    combined_scores: Dict[int, Dict] = {}

    # Semantic scores normalize karo (already 0-1 range mein hain, cosine similarity se)
    for r in semantic_results:
        chunk_id = r["chunk_id"]
        combined_scores[chunk_id] = {
            **r,
            "combined_score": r["score"] * semantic_weight
        }

    # BM25 scores normalize karo (max score se divide karke 0-1 range mein lao)
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

    # Combined score ke hisaab se sort karo
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