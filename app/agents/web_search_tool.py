
"""
web_search_tool.py
--------------------
Tavily API se web search karta hai jab document mein answer na mile.

Tavily ek LLM-friendly search API hai — ye Google jaisa raw HTML nahi
deta, balki already-summarized, clean snippets deta hai jo directly
LLM ko context ke roop mein diye ja sakte hain.
"""

from tavily import TavilyClient
from typing import List, Dict
from app.config import settings

_client = None


def get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    return _client


def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Query leta hai aur web se relevant results laata hai.

    Returns: [{"title": "...", "content": "...", "url": "..."}, ...]
    Ye format vector_store.search() ke output jaisa rakha hai
    (text + source) taaki downstream code same tarah handle kar sake.
    """
    client = get_client()

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic"  # "advanced" zyada accurate hai but slow + costly
        )
    except Exception as e:
        raise ValueError(f"Web search failed: {str(e)}")

    results = []
    for item in response.get("results", []):
        results.append({
            "text": item.get("content", ""),
            "source": item.get("title", "Unknown source"),
            "url": item.get("url", ""),
            "score": item.get("score", 0.0)
        })

    return results


# --- Quick test ---
if __name__ == "__main__":
    results = web_search("latest AI news July 2026")
    print(f"Found {len(results)} results")
    for r in results[:2]:
        print(f"\nSource: {r['source']}")
        print(f"URL: {r['url']}")
        print(f"Content preview: {r['text'][:200]}")