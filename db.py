"""
Database layer for StartupLex.
Manages users, chat sessions, and chat messages using SQLite.
"""
import os
import sqlite3
import uuid
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.getenv('DATABASE_PATH', 'startuplex_users.db')


def _get_connection():
    """Create and return a new database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create all tables if they do not exist."""
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            title TEXT,
            created_at TEXT
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES chat_sessions(id),
            role TEXT,
            content TEXT,
            created_at TEXT
        )
    ''')

    conn.commit()
    conn.close()


# ──────────────────────────────── User Operations ────────────────────────────────

def create_user(name, email, password):
    """Create a new user. Returns user dict or None if email already exists."""
    password_hash = generate_password_hash(password)
    created_at = datetime.now().isoformat()
    conn = _get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            'INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)',
            (name, email, password_hash, created_at)
        )
        conn.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        return None
    conn.close()
    return {'id': user_id, 'name': name, 'email': email, 'created_at': created_at}


def get_user_by_email(email):
    """Retrieve a user by email. Returns dict or None."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE email = ?', (email,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id):
    """Retrieve a user by ID. Returns dict or None."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def verify_password(stored_hash, password):
    """Verify a password against its hash."""
    return check_password_hash(stored_hash, password)


# ──────────────────────────── Chat Session Operations ────────────────────────────

def create_chat_session(user_id, title):
    """Create a new chat session. Returns session dict."""
    session_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO chat_sessions (id, user_id, title, created_at) VALUES (?, ?, ?, ?)',
        (session_id, user_id, title, created_at)
    )
    conn.commit()
    conn.close()
    return {'id': session_id, 'user_id': user_id, 'title': title, 'created_at': created_at}


def get_user_sessions(user_id):
    """Retrieve all chat sessions for a user, ordered by most recent first."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        'SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY created_at DESC',
        (user_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_session_messages(session_id):
    """Retrieve all messages for a chat session, ordered chronologically."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        'SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC',
        (session_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_chat_message(session_id, role, content):
    """Save a chat message. Returns message dict."""
    created_at = datetime.now().isoformat()
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)',
        (session_id, role, content, created_at)
    )
    conn.commit()
    message_id = cur.lastrowid
    conn.close()
    return {'id': message_id, 'session_id': session_id, 'role': role, 'content': content, 'created_at': created_at}


def delete_chat_session(session_id):
    """Delete a chat session and all its messages."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
    cur.execute('DELETE FROM chat_sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()


def update_session_title(session_id, title):
    """Update the title of a chat session."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute('UPDATE chat_sessions SET title = ? WHERE id = ?', (title, session_id))
    conn.commit()
    conn.close()
