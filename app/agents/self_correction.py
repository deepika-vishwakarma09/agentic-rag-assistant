
"""
self_correction.py
---------------------
Answer generate hone ke baad, ye module use kaam karta hai:
1. Verify karta hai ki answer context se supported hai ya nahi
2. Agar nahi (hallucination lagti hai), to ek strict retry karta hai

Ye "reflection agent" pattern hai — agent apna khud ka output check
karta hai aur improve karne ki koshish karta hai, bina human ke.
"""

from app.generation.llm_client import call_llm
from app.generation.prompts import SELF_CORRECTION_PROMPT, STRICT_RETRY_PROMPT


def verify_answer(context: str, answer: str) -> bool:
    """
    Answer ko context ke against check karta hai.
    Returns: True agar valid (context-supported), False agar invalid (hallucination lagti hai)
    """
    prompt = SELF_CORRECTION_PROMPT.format(context=context, answer=answer)

    verdict = call_llm(
        system_prompt="You are a strict fact-checker.",
        user_prompt=prompt,
        temperature=0.0  # verification deterministic honi chahiye
    ).lower().strip()

    return "valid" in verdict and "invalid" not in verdict


def retry_with_strict_grounding(context: str, question: str) -> str:
    """
    Jab pehla answer invalid nikle, isse ek zyada strict, careful
    answer generate hota hai jo sirf context pe strictly based ho.
    """
    prompt = STRICT_RETRY_PROMPT.format(context=context, question=question)

    return call_llm(
        system_prompt="You are a careful, strictly accurate assistant.",
        user_prompt=prompt,
        temperature=0.1  # aur bhi kam creativity, zyada factual
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