import uuid
import sqlite3
from database.db import Database
from config.settings import settings


# ===== INIT DB =====
db = Database(settings.DB_PATH)


# =========================
# USER
# =========================
def create_user(name: str) -> str:
    conn = db.get_conn()
    user_id = str(uuid.uuid4())

    with conn:
        conn.execute(
            "INSERT INTO users (id, name) VALUES (?, ?)",
            (user_id, name)
        )

    return user_id


def get_user(user_id: str):
    conn = db.get_conn()
    cur = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,)
    )
    return cur.fetchone()


# =========================
# FILE
# =========================
def save_file(user_id: str, file_name: str, path: str) -> str:
    conn = db.get_conn()
    file_id = str(uuid.uuid4())

    with conn:
        conn.execute("""
            INSERT INTO files (id, user_id, file_name, path)
            VALUES (?, ?, ?, ?)
        """, (file_id, user_id, file_name, path))

    return file_id


def get_user_files(user_id: str):
    conn = db.get_conn()
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


def save_chat(user_id: str, session_id: str, role: str, content: str):
    conn = db.get_conn()

    with conn:
        conn.execute("""
            INSERT INTO chats (session_id, user_id, role, content)
            VALUES (?, ?, ?, ?)
        """, (session_id, user_id, role, content))


def get_chat_history(user_id: str, session_id: str, limit: int = 20):
    conn = db.get_conn()

    cur = conn.execute("""
        SELECT role, content FROM chats
        WHERE user_id = ? AND session_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, session_id, limit))

    rows = cur.fetchall()
    return rows[::-1]