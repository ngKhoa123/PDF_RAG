from collections import deque
from typing import List, Tuple
import uuid
import time


class SessionMemory:
    """
    In-memory session buffer (short-term memory)
    - Stores recent turns only
    - Each turn = (user, assistant)
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.memory = deque(maxlen=max_turns * 2)  # user + assistant

    # =========================
    # ADD TURN (atomic)
    # =========================
    def add_turn(self, user_msg: str, ai_msg: str):
        turn_id = str(uuid.uuid4())
        timestamp = time.time()

        self.memory.append(("user", user_msg, turn_id, timestamp))
        self.memory.append(("assistant", ai_msg, turn_id, timestamp))

    # =========================
    # GET RECENT 
    # =========================
    def get_recent(self, k: int = 3) -> List[Tuple[str, str]]:
        items = list(self.memory)[-k * 2:]
        return [(role, content) for role, content, _, _ in items]

    # =========================
    # GET ALL (optional)
    # =========================
    def get(self) -> List[Tuple[str, str]]:
        return [(role, content) for role, content, _, _ in self.memory]

    # =========================
    # CLEAR
    # =========================
    def clear(self):
        self.memory.clear()