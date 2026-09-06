import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Dict, Union

from core.interfaces.memory_interface import MemoryServiceInterface
from core.models.memory import ChatMessage, MemoryEntry, MessageRole


DEFAULT_STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
DEFAULT_DB_PATH = DEFAULT_STORAGE_DIR / "memory.db"


class SQLiteMemoryStore(MemoryServiceInterface):
    """
    Persistent, thread-safe, anchored SQLite implementation of MemoryServiceInterface.
    Stores session-isolated conversation history (conversations) and user-scoped
    preferences/facts (preferences).
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Initialize SQLiteMemoryStore.

        Args:
            db_path: Explicit path to the SQLite database file. Defaults to
                     'backend/storage/memory.db' if omitted.
        """
        if db_path is not None:
            self.db_path = Path(db_path).resolve()
        else:
            self.db_path = DEFAULT_DB_PATH.resolve()

        # Ensure target storage directory exists safely
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_session_time
                ON conversations (session_id, timestamp);
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, key)
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_preferences_user
                ON preferences (user_id);
            """)
            conn.commit()

    # ========================================================================
    # CONVERSATION HISTORY (EPISODIC MEMORY)
    # ========================================================================

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
    ) -> ChatMessage:
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        if not isinstance(role, MessageRole):
            raise TypeError(f"role must be a MessageRole enum, got {type(role).__name__}")
        if not isinstance(content, str):
            raise TypeError(f"content must be a string, got {type(content).__name__}")

        msg_id = message_id if (message_id and isinstance(message_id, str)) else f"msg-{uuid.uuid4().hex[:12]}"
        now = datetime.now()
        meta = metadata if metadata is not None else {}
        meta_json = json.dumps(meta, sort_keys=True)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (msg_id, session_id.strip(), role.value, content, now.isoformat(), meta_json),
            )
            conn.commit()

        return ChatMessage(
            id=msg_id,
            session_id=session_id.strip(),
            role=role,
            content=content,
            timestamp=now,
            metadata=meta,
        )

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[ChatMessage]:
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")

        if limit is not None and limit <= 0:
            return []

        sid = session_id.strip()

        with self._get_connection() as conn:
            if limit is None:
                cursor = conn.execute(
                    """
                    SELECT id, session_id, role, content, timestamp, metadata
                    FROM conversations
                    WHERE session_id = ?
                    ORDER BY timestamp ASC, rowid ASC
                    """,
                    (sid,),
                )
                rows = cursor.fetchall()
            else:
                cursor = conn.execute(
                    """
                    SELECT id, session_id, role, content, timestamp, metadata
                    FROM (
                        SELECT id, session_id, role, content, timestamp, metadata, rowid
                        FROM conversations
                        WHERE session_id = ?
                        ORDER BY timestamp DESC, rowid DESC
                        LIMIT ?
                    )
                    ORDER BY timestamp ASC, rowid ASC
                    """,
                    (sid, limit),
                )
                rows = cursor.fetchall()

        messages: List[ChatMessage] = []
        for row in rows:
            mid, s_id, r_val, text, ts_str, meta_str = row
            try:
                role = MessageRole(r_val)
            except ValueError:
                role = MessageRole.USER
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                ts = datetime.now()
            try:
                meta = json.loads(meta_str) if meta_str else {}
            except json.JSONDecodeError:
                meta = {}

            messages.append(
                ChatMessage(
                    id=mid,
                    session_id=s_id,
                    role=role,
                    content=text,
                    timestamp=ts,
                    metadata=meta,
                )
            )

        return messages

    def clear_session(self, session_id: str) -> None:
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")

        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE session_id = ?",
                (session_id.strip(),),
            )
            conn.commit()

    def pop_last_message(self, session_id: str) -> Optional[ChatMessage]:
        """
        Delete and return the most recent message in the specified session, or None if empty.
        """
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")

        sid = session_id.strip()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, session_id, role, content, timestamp, metadata
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp DESC, rowid DESC
                LIMIT 1
                """,
                (sid,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            mid, s_id, r_val, text, ts_str, meta_str = row
            conn.execute("DELETE FROM conversations WHERE id = ?", (mid,))
            conn.commit()

        try:
            role = MessageRole(r_val)
        except ValueError:
            role = MessageRole.USER
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            ts = datetime.now()
        try:
            meta = json.loads(meta_str) if meta_str else {}
        except json.JSONDecodeError:
            meta = {}

        return ChatMessage(
            id=mid,
            session_id=s_id,
            role=role,
            content=text,
            timestamp=ts,
            metadata=meta,
        )

    # ========================================================================
    # PREFERENCE & FACT STORAGE (PERSONAL MEMORY)
    # ========================================================================

    def save_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
        category: str = "general",
    ) -> MemoryEntry:
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")
        if not key or not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string.")
        if not category or not isinstance(category, str) or not category.strip():
            raise ValueError("category must be a non-empty string.")

        uid = user_id.strip()
        k = key.strip()
        cat = category.strip()
        now = datetime.now()
        val_json = json.dumps(value, sort_keys=True)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO preferences (user_id, key, value, category, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (uid, k, val_json, cat, now.isoformat()),
            )
            conn.commit()

        return MemoryEntry(
            key=k,
            value=value,
            category=cat,
            user_id=uid,
            updated_at=now,
        )

    def get_preference(
        self,
        user_id: str,
        key: str,
    ) -> Optional[MemoryEntry]:
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")
        if not key or not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string.")

        uid = user_id.strip()
        k = key.strip()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT user_id, key, value, category, updated_at
                FROM preferences
                WHERE user_id = ? AND key = ?
                """,
                (uid, k),
            )
            row = cursor.fetchone()

        if not row:
            return None

        u_id, entry_key, val_json, cat, ts_str = row
        try:
            val = json.loads(val_json)
        except (json.JSONDecodeError, TypeError):
            val = val_json

        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            ts = datetime.now()

        return MemoryEntry(
            key=entry_key,
            value=val,
            category=cat,
            user_id=u_id,
            updated_at=ts,
        )

    def delete_preference(
        self,
        user_id: str,
        key: str,
    ) -> bool:
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")
        if not key or not isinstance(key, str) or not key.strip():
            raise ValueError("key must be a non-empty string.")

        uid = user_id.strip()
        k = key.strip()

        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM preferences WHERE user_id = ? AND key = ?",
                (uid, k),
            )
            conn.commit()
            return cursor.rowcount > 0

    def list_preferences(
        self,
        user_id: str,
    ) -> List[MemoryEntry]:
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")

        uid = user_id.strip()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT user_id, key, value, category, updated_at
                FROM preferences
                WHERE user_id = ?
                ORDER BY key ASC
                """,
                (uid,),
            )
            rows = cursor.fetchall()

        entries: List[MemoryEntry] = []
        for row in rows:
            u_id, entry_key, val_json, cat, ts_str = row
            try:
                val = json.loads(val_json)
            except (json.JSONDecodeError, TypeError):
                val = val_json

            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                ts = datetime.now()

            entries.append(
                MemoryEntry(
                    key=entry_key,
                    value=val,
                    category=cat,
                    user_id=u_id,
                    updated_at=ts,
                )
            )
        return entries
