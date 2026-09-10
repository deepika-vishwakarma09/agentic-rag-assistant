
"""
conversation_store.py
------------------------
Manages chat history, session-wise.

Why sessions? Because if multiple users/tabs are chatting at the same time,
each one's history should be kept separate, not mixed together.

For now, we are storing in-memory (Python dict) — simple and sufficient
for a demo/project. In production, this would use Redis or a database
(so that history is not lost on server restart).
"""

from typing import List, Dict


class ConversationStore:
    def __init__(self):
        # session_id -> list of {"role": "user"/"assistant", "content": "..."}
        self._sessions: Dict[str, List[Dict]] = {}

    def add_message(self, session_id: str, role: str, content: str):
        """Add a new message to the history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append({
            "role": role,
            "content": content
        })

    def get_history(self, session_id: str, last_n: int = 5) -> List[Dict]:
        """
        Returns the session's history (default: last 5 messages).

        Why do we limit last_n? Because sending the entire history to the LLM
        becomes costly and slow in long conversations —
        recent context is usually enough to understand follow-ups.
        """
        return self._sessions.get(session_id, [])[-last_n:]

    def get_history_as_text(self, session_id: str, last_n: int = 5) -> str:
        """
        Formats the history into a readable string, ready to be
        directly inserted into a prompt.
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
        """Clear a session's history (to start a new conversation)."""
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