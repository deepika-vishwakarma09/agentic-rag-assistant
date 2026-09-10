
"""
llm_client.py
--------------
Wrapper for calling the LLaMA model via the Groq API.

Why Groq? Because it provides very fast inference (uses LPU hardware)
and the free tier is generous for fresher/student projects.
"""

from groq import Groq
from app.config import settings

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """
    Sends a system prompt and user prompt to the LLM and gets a response.

    Temperature is kept low (0.3) because we need factual, consistent answers
    — not creative writing.
    """
    client = get_client()

    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=1024
    )

    return response.choices[0].message.content.strip()


# --- Quick test ---
if __name__ == "__main__":
    result = call_llm(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is 2+2? Answer in one word."
    )
    print(f"LLM response: {result}")