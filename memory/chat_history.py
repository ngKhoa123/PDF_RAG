import sqlite3
from typing import List, Tuple
from config.settings import settings


class ChatHistory:
    def __init__(self, db_path: str = settings.DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    # =========================
    # INIT TABLE
    # =========================
    def _create_table(self):
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                role TEXT,
                content TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)

            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_user
            ON chat_history(user_id)
            """)

    # =========================
    # SAVE
    # =========================
    def add(self, user_id: str, role: str, content: str):
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO chat_history (user_id, role, content)
                VALUES (?, ?, ?)
                """,
                (user_id, role, content)
            )

    # =========================
    # LOAD
    # =========================
    def load(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[Tuple[str, str]]:

        cur = self.conn.execute("""
            SELECT role, content FROM chat_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (user_id, limit))

        rows = cur.fetchall()
        return [(row["role"], row["content"]) for row in rows[::-1]]

    # =========================
    # CLEAR USER
    # =========================
    def clear(self, user_id: str):
        with self.conn:
            self.conn.execute(
                "DELETE FROM chat_history WHERE user_id = ?",
                (user_id,)
            )

    # =========================
    # CLOSE
    # =========================
    def close(self):
        self.conn.close()