
"""
conversation_store.py
------------------------
Chat history manage karta hai, session-wise.

Session kyun? Kyunki agar multiple users/tabs ek saath chat kar rahe hon,
to har ek ki history alag rehni chahiye, mix nahi honi chahiye.

Abhi ke liye in-memory (Python dict) store kar rahe hain — simple aur
demo/project ke liye kaafi hai. Production mein ye Redis ya database
mein hota (server restart hone par history na ude).
"""

from typing import List, Dict


class ConversationStore:
    def __init__(self):
        # session_id -> list of {"role": "user"/"assistant", "content": "..."}
        self._sessions: Dict[str, List[Dict]] = {}

    def add_message(self, session_id: str, role: str, content: str):
        """Ek naya message history mein jodo."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append({
            "role": role,
            "content": content
        })

    def get_history(self, session_id: str, last_n: int = 5) -> List[Dict]:
        """
        Session ki history return karta hai (default: last 5 messages).

        last_n kyun limit karte hain? Kyunki poori history LLM ko
        bhejna costly aur slow ho jaata hai lambi conversations mein —
        recent context usually kaafi hota hai follow-ups samajhne ke liye.
        """
        return self._sessions.get(session_id, [])[-last_n:]

    def get_history_as_text(self, session_id: str, last_n: int = 5) -> str:
        """
        History ko ek readable string mein format karta hai, prompt mein
        directly daalne ke liye.
        """
        history = self.get_history(session_id, last_n)
        if not history:
            return "No previous conversation."

        lines = []
        for msg in history:
            speaker = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {msg['content']}")

        return "\n".join(lines)

    def clear_session(self, session_id: str):
        """Session ki history clear karo (naya conversation shuru karne ke liye)."""
        if session_id in self._sessions:
            del self._sessions[session_id]


# --- Quick test ---
if __name__ == "__main__":
    store = ConversationStore()

    store.add_message("session_1", "user", "What is the refund policy?")
    store.add_message("session_1", "assistant", "Refunds are allowed within 30 days.")
    store.add_message("session_1", "user", "What about after that?")

    print("History for session_1:")
    print(store.get_history_as_text("session_1"))

    print("\nHistory for a fresh session_2 (should be empty):")
    print(store.get_history_as_text("session_2"))