"""
Compatibility shim for legacy V1 and Phase 2 cutover bridge callers.
Delegates conversation storage directly to SQLiteMemoryStore instead of
maintaining an isolated, unpersisted in-memory list.
"""

from typing import List, Dict, Any, Optional
from core.models.memory import MessageRole
from memory.sqlite_store import SQLiteMemoryStore


MAX_HISTORY = 20
DEFAULT_SESSION_ID = "default"

_store: Optional[SQLiteMemoryStore] = None


def get_store() -> SQLiteMemoryStore:
    """Retrieve or lazily initialize the singleton SQLiteMemoryStore instance."""
    global _store
    if _store is None:
        _store = SQLiteMemoryStore()
    return _store


def set_store(store: SQLiteMemoryStore) -> None:
    """Explicitly inject a custom SQLiteMemoryStore instance (primarily for testing)."""
    global _store
    _store = store


class HistoryList(list):
    """
    Subclass of list returned by get_history() to preserve 100% backward
    compatibility with callers that pop or slice conversation history (such as
    assistant.py pre-execution planning rollback).
    """

    def __init__(
        self,
        items: List[Dict[str, str]],
        store: SQLiteMemoryStore,
        session_id: str = DEFAULT_SESSION_ID,
    ):
        super().__init__(items)
        self._store = store
        self._session_id = session_id

    def pop(self, index: int = -1) -> Dict[str, str]:
        item = super().pop(index)
        # If popping the latest item, remove it from the backing SQLite store
        if index in (-1, len(self)):
            self._store.pop_last_message(self._session_id)
        return item


def add_message(role: str, content: str) -> None:
    """
    Add a message to the default conversation history session.
    Delegates to the backing SQLiteMemoryStore.
    """
    role_str = str(role).lower().strip()
    try:
        msg_role = MessageRole(role_str)
    except ValueError:
        msg_role = MessageRole.USER

    store = get_store()
    store.add_message(
        session_id=DEFAULT_SESSION_ID,
        role=msg_role,
        content=str(content),
    )


def get_history() -> HistoryList:
    """
    Retrieve recent conversation history for the default session as a list of dicts.
    Returns a HistoryList instance bounded by MAX_HISTORY in chronological order.
    """
    store = get_store()
    messages = store.get_history(session_id=DEFAULT_SESSION_ID, limit=MAX_HISTORY)
    items = [{"role": m.role.value, "content": m.content} for m in messages]
    return HistoryList(items, store=store, session_id=DEFAULT_SESSION_ID)


def clear_history() -> None:
    """
    Clear all messages in the default conversation history session.
    """
    store = get_store()
    store.clear_session(DEFAULT_SESSION_ID)