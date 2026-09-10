
"""
self_correction.py
---------------------
After an answer is generated, this module does the following:
1. Verifies whether the answer is supported by the context or not
2. If not (hallucination is suspected), it performs a strict retry

This is the "reflection agent" pattern — the agent checks its own output
and tries to improve it, without human intervention.
"""

from app.generation.llm_client import call_llm
from app.generation.prompts import SELF_CORRECTION_PROMPT, STRICT_RETRY_PROMPT


def verify_answer(context: str, answer: str) -> bool:
    """
    Checks the answer against the context.
    Returns: True if valid (context-supported), False if invalid (hallucination suspected)
    """
    prompt = SELF_CORRECTION_PROMPT.format(context=context, answer=answer)

    verdict = call_llm(
        system_prompt="You are a strict fact-checker.",
        user_prompt=prompt,
        temperature=0.0  # verification should be deterministic
    ).lower().strip()

    return "valid" in verdict and "invalid" not in verdict


def retry_with_strict_grounding(context: str, question: str) -> str:
    """
    When the first answer turns out to be invalid, this generates a more
    strict, careful answer that is strictly based only on the context.
    """
    prompt = STRICT_RETRY_PROMPT.format(context=context, question=question)

    return call_llm(
        system_prompt="You are a careful, strictly accurate assistant.",
        user_prompt=prompt,
        temperature=0.1  # even less creativity, more factual
    )


# --- Quick test ---
if __name__ == "__main__":
    context = "The refund policy allows returns within 30 days of purchase."

    good_answer = "You can return items within 30 days of purchase."
    bad_answer = "You can return items within 30 days and also get free shipping forever."

    print(f"Good answer valid? {verify_answer(context, good_answer)}")
    print(f"Bad answer valid? {verify_answer(context, bad_answer)}")

    if not verify_answer(context, bad_answer):
        corrected = retry_with_strict_grounding(context, "What is the refund policy?")
        print(f"\nCorrected answer: {corrected}")