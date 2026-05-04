import sqlite3
import uuid
from typing import List, Optional
from config.settings import settings


class Database:
    def __init__(self, db_path: str = settings.DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_tables()

    def _init_tables(self):
        with self.conn:
            # ===== USERS =====
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT
            )
            """)

            # ===== FILES =====
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                file_name TEXT,
                path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """)

            # ===== CHATS =====
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                role TEXT,
                content TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """)

            # ===== INDEX =====
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_user_id ON files(user_id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id)"
            )

    def get_conn(self):
        return self.conn

    def close(self):
        self.conn.close()


# =========================
# USER
# =========================
def create_user(conn: sqlite3.Connection, name: str) -> str:
    user_id = str(uuid.uuid4())
    with conn:
        conn.execute(
            "INSERT INTO users (id, name) VALUES (?, ?)",
            (user_id, name)
        )
    return user_id


def get_user(conn: sqlite3.Connection, user_id: str):
    cur = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cur.fetchone()


# =========================
# FILE
# =========================
def save_file(conn: sqlite3.Connection, user_id: str, file_name: str, path: str) -> str:
    file_id = str(uuid.uuid4())
    with conn:
        conn.execute("""
            INSERT INTO files (id, user_id, file_name, path)
            VALUES (?, ?, ?, ?)
        """, (file_id, user_id, file_name, path))
    return file_id


def get_user_files(conn: sqlite3.Connection, user_id: str):
    cur = conn.execute(
        "SELECT * FROM files WHERE user_id = ?",
        (user_id,)
    )
    return cur.fetchall()


# =========================
# CHAT
# =========================
def create_session() -> str:
    return str(uuid.uuid4())


def save_chat(
    conn: sqlite3.Connection,
    user_id: str,
    session_id: str,
    role: str,
    content: str
):
    with conn:
        conn.execute("""
            INSERT INTO chats (session_id, user_id, role, content)
            VALUES (?, ?, ?, ?)
        """, (session_id, user_id, role, content))


def get_chat_history(
    conn: sqlite3.Connection,
    user_id: str,
    session_id: str,
    limit: int = 20
):
    cur = conn.execute("""
        SELECT role, content FROM chats
        WHERE user_id = ? AND session_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, session_id, limit))

    rows = cur.fetchall()
    return rows[::-1]  # reverse → oldest first