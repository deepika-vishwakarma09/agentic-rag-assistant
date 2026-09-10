
"""
web_search_tool.py
--------------------
Performs web search using the Tavily API when the answer is not found in the document.

Tavily is an LLM-friendly search API — it doesn't return raw HTML like Google,
but instead returns already-summarized, clean snippets that can be directly
provided to the LLM as context.
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
    Takes a query and fetches relevant results from the web.

    Returns: [{"title": "...", "content": "...", "url": "..."}, ...]
    This format is kept similar to vector_store.search() output
    (text + source) so that downstream code can handle both the same way.
    """
    client = get_client()

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic"  # "advanced" is more accurate but slow + costly
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