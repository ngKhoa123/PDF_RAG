from memory.chat_history import ChatHistory
from memory.session_memory import SessionMemory


class MemoryManager:
    """
    Hybrid memory:
    - SessionMemory (fast, short-term)
    - ChatHistory (persistent fallback)
    """

    def __init__(self):
        self.chat_db = ChatHistory()
        self.sessions = {}

    # =========================
    # SESSION
    # =========================
    def _get_session(self, user_id):
        if user_id not in self.sessions:
            self.sessions[user_id] = SessionMemory()
        return self.sessions[user_id]

    # =========================
    # ADD TURN
    # =========================
    def add_turn(self, user_id, user_msg, ai_msg):
        session = self._get_session(user_id)

        # session memory
        session.add_turn(user_msg, ai_msg)

        # persistent DB
        self.chat_db.add(user_id, "user", user_msg)
        self.chat_db.add(user_id, "assistant", ai_msg)

    # =========================
    # GET CONTEXT (🔥 CORE)
    # =========================
    def get_context(self, user_id, max_turns: int = 3) -> str:
        session = self._get_session(user_id)

        # 1. lấy recent từ RAM
        history = session.get_recent(max_turns)

        # 2. fallback DB nếu rỗng
        if not history:
            history = self.chat_db.load(user_id, limit=max_turns * 2)

        return self._format(history)

    # =========================
    # FORMAT
    # =========================
    def _format(self, history):
        if not history:
            return ""

        lines = []

        for role, content in history:
            if role == "user":
                lines.append(f"User: {content}")
            else:
                lines.append(f"Assistant: {content}")

        return "\n".join(lines)

    # =========================
    # CLEAR
    # =========================
    def clear(self, user_id):
        if user_id in self.sessions:
            self.sessions[user_id].clear()

        self.chat_db.clear(user_id)