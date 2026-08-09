
"""
llm_client.py
--------------
Groq API se LLaMA model ko call karne ka wrapper.

Groq kyun? Kyunki ye bahut fast inference deta hai (LPU hardware use karta hai)
aur free tier generous hai fresher projects ke liye.
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
    LLM ko ek system prompt aur user prompt deke response leta hai.

    temperature kam (0.3) rakha hai kyunki hume factual, consistent answers
    chahiye — creative writing nahi.
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