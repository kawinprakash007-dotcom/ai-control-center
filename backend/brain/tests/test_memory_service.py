import os
import sqlite3
from datetime import datetime
from pathlib import Path
import pytest

from core.models.memory import MessageRole, ChatMessage, MemoryEntry
from core.interfaces.memory_interface import MemoryServiceInterface, MemoryInterface
from memory.sqlite_store import SQLiteMemoryStore, DEFAULT_DB_PATH
import memory.history as history_shim


@pytest.fixture
def temp_store(tmp_path):
    """Provide an isolated, deterministic SQLiteMemoryStore instance for testing."""
    db_file = tmp_path / "test_memory.db"
    return SQLiteMemoryStore(db_path=db_file)


# ============================================================================
# 1. DOMAIN MODEL TESTS
# ============================================================================

def test_chat_message_construction():
    """1. ChatMessage can be cleanly instantiated and fields are stored immutably."""
    now = datetime.now()
    meta = {"source": "chat", "tokens": 12}
    msg = ChatMessage(
        id="msg-001",
        session_id="session-alpha",
        role=MessageRole.USER,
        content="Hello, system!",
        timestamp=now,
        metadata=meta,
    )

    assert msg.id == "msg-001"
    assert msg.session_id == "session-alpha"
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello, system!"
    assert msg.timestamp == now
    assert msg.metadata == {"source": "chat", "tokens": 12}

    # Metadata dictionary isolation
    meta["tokens"] = 99
    assert msg.metadata["tokens"] == 12

    # Frozen immutability
    with pytest.raises(Exception):
        msg.content = "New content"


def test_message_role_values():
    """2. MessageRole enum defines exactly USER, ASSISTANT, SYSTEM, and TOOL."""
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"
    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.TOOL.value == "tool"
    assert len(MessageRole) == 4


def test_memory_entry_construction():
    """3. MemoryEntry can be cleanly instantiated and fields are stored immutably."""
    now = datetime.now()
    entry = MemoryEntry(
        key="editor",
        value="neovim",
        category="developer_tools",
        user_id="user-42",
        updated_at=now,
    )

    assert entry.key == "editor"
    assert entry.value == "neovim"
    assert entry.category == "developer_tools"
    assert entry.user_id == "user-42"
    assert entry.updated_at == now

    # Frozen immutability
    with pytest.raises(Exception):
        entry.value = "vscode"


# ============================================================================
# 2. CONVERSATION HISTORY & EPISODIC STORAGE TESTS
# ============================================================================

def test_empty_history(temp_store):
    """4. Querying history for a non-existent session returns an empty list."""
    history = temp_store.get_history(session_id="non-existent")
    assert history == []
    assert isinstance(history, list)


def test_add_message(temp_store):
    """5. Adding a message stores it in SQLite and returns a valid ChatMessage."""
    msg = temp_store.add_message(
        session_id="sess-1",
        role=MessageRole.USER,
        content="How does Linux virtual memory work?",
        metadata={"priority": "high"},
    )

    assert isinstance(msg, ChatMessage)
    assert msg.session_id == "sess-1"
    assert msg.role == MessageRole.USER
    assert msg.content == "How does Linux virtual memory work?"
    assert msg.metadata == {"priority": "high"}

    history = temp_store.get_history(session_id="sess-1")
    assert len(history) == 1
    assert history[0].id == msg.id
    assert history[0].content == "How does Linux virtual memory work?"


def test_chronological_history_ordering(temp_store):
    """6. Multiple messages are returned in exact chronological sequence."""
    temp_store.add_message("sess-seq", MessageRole.USER, "First")
    temp_store.add_message("sess-seq", MessageRole.ASSISTANT, "Second")
    temp_store.add_message("sess-seq", MessageRole.USER, "Third")
    temp_store.add_message("sess-seq", MessageRole.ASSISTANT, "Fourth")

    history = temp_store.get_history("sess-seq")
    assert len(history) == 4
    contents = [m.content for m in history]
    assert contents == ["First", "Second", "Third", "Fourth"]


def test_history_limit_returns_latest_n_messages(temp_store):
    """7. History limit returns the latest N messages in chronological order."""
    for i in range(1, 11):
        temp_store.add_message("sess-limit", MessageRole.USER, f"Msg {i}")

    # Ask for latest 3
    history = temp_store.get_history("sess-limit", limit=3)
    assert len(history) == 3
    assert [m.content for m in history] == ["Msg 8", "Msg 9", "Msg 10"]

    # Limit <= 0 returns empty list without error
    assert temp_store.get_history("sess-limit", limit=0) == []
    assert temp_store.get_history("sess-limit", limit=-5) == []


def test_session_isolation(temp_store):
    """8. Messages in session-A are strictly isolated from session-B."""
    temp_store.add_message("session-A", MessageRole.USER, "Message for A")
    temp_store.add_message("session-B", MessageRole.USER, "Message for B")

    history_a = temp_store.get_history("session-A")
    history_b = temp_store.get_history("session-B")

    assert len(history_a) == 1
    assert history_a[0].content == "Message for A"

    assert len(history_b) == 1
    assert history_b[0].content == "Message for B"


def test_persistence_across_separate_connections(tmp_path):
    """9. Data written in one store instance persists across separate connections."""
    db_file = tmp_path / "persist.db"

    # Store 1 writes data and finishes
    store1 = SQLiteMemoryStore(db_path=db_file)
    store1.add_message("sess-p", MessageRole.USER, "Persistent message")
    store1.save_preference("user-p", "shell", "zsh")
    del store1

    # Store 2 opens the same database file and reads data
    store2 = SQLiteMemoryStore(db_path=db_file)
    history = store2.get_history("sess-p")
    assert len(history) == 1
    assert history[0].content == "Persistent message"

    pref = store2.get_preference("user-p", "shell")
    assert pref is not None
    assert pref.value == "zsh"


def test_clear_session(temp_store):
    """10. Clear session purges only the specified session messages."""
    temp_store.add_message("sess-1", MessageRole.USER, "Keep 1")
    temp_store.add_message("sess-2", MessageRole.USER, "Purge 2")

    temp_store.clear_session("sess-2")

    assert len(temp_store.get_history("sess-1")) == 1
    assert len(temp_store.get_history("sess-2")) == 0


def test_clear_session_does_not_remove_preferences(temp_store):
    """11. Clearing a conversation session does not delete stored preferences."""
    temp_store.add_message("sess-mix", MessageRole.USER, "Hello")
    temp_store.save_preference("user-mix", "theme", "nord")

    temp_store.clear_session("sess-mix")

    assert len(temp_store.get_history("sess-mix")) == 0
    pref = temp_store.get_preference("user-mix", "theme")
    assert pref is not None
    assert pref.value == "nord"


# ============================================================================
# 3. PREFERENCES & FACT STORAGE (PERSONAL MEMORY) TESTS
# ============================================================================

def test_save_and_get_preference(temp_store):
    """12 & 13. Save and retrieve user-scoped preference."""
    entry = temp_store.save_preference("user-1", "editor", "vscode", category="tools")
    assert entry.key == "editor"
    assert entry.value == "vscode"
    assert entry.category == "tools"
    assert entry.user_id == "user-1"

    fetched = temp_store.get_preference("user-1", "editor")
    assert fetched is not None
    assert fetched.key == "editor"
    assert fetched.value == "vscode"
    assert fetched.category == "tools"
    assert fetched.user_id == "user-1"

    # Non-existent preference returns None
    assert temp_store.get_preference("user-1", "non_existent_key") is None


def test_update_preference(temp_store):
    """14. Updating a preference replaces the value without creating duplicates."""
    temp_store.save_preference("user-update", "color", "blue")
    temp_store.save_preference("user-update", "color", "green")

    fetched = temp_store.get_preference("user-update", "color")
    assert fetched is not None
    assert fetched.value == "green"


def test_user_isolation_for_preferences(temp_store):
    """15. Preferences with the same key remain isolated across different users."""
    temp_store.save_preference("user-alice", "editor", "emacs")
    temp_store.save_preference("user-bob", "editor", "vim")

    alice_pref = temp_store.get_preference("user-alice", "editor")
    bob_pref = temp_store.get_preference("user-bob", "editor")

    assert alice_pref is not None and alice_pref.value == "emacs"
    assert bob_pref is not None and bob_pref.value == "vim"


def test_structured_json_preference_values(temp_store):
    """16. Complex JSON structures (dicts, lists, bools, numbers) round-trip properly."""
    config = {
        "notifications": True,
        "ports": [8000, 8080],
        "sub_config": {"retries": 3, "timeout": 15.5},
    }
    temp_store.save_preference("user-json", "runtime_config", config)

    entry = temp_store.get_preference("user-json", "runtime_config")
    assert entry is not None
    assert entry.value == config
    assert entry.value["notifications"] is True
    assert entry.value["ports"] == [8000, 8080]
    assert entry.value["sub_config"]["timeout"] == 15.5


def test_metadata_round_trip(temp_store):
    """17. ChatMessage metadata round-trips correctly through SQLite JSON."""
    meta = {"source": "voice", "confidence": 0.98, "entities": ["notepad", "time"]}
    msg = temp_store.add_message("sess-meta", MessageRole.USER, "Open Notepad", metadata=meta)

    history = temp_store.get_history("sess-meta")
    assert len(history) == 1
    assert history[0].metadata == meta
    assert history[0].metadata["confidence"] == 0.98


def test_delete_preference(temp_store):
    """Delete preference removes only the targeted key and returns True/False."""
    temp_store.save_preference("user-del", "k1", "v1")
    temp_store.save_preference("user-del", "k2", "v2")

    # Deleting existing key returns True
    assert temp_store.delete_preference("user-del", "k1") is True
    assert temp_store.get_preference("user-del", "k1") is None

    # Other key remains untouched
    assert temp_store.get_preference("user-del", "k2") is not None
    assert temp_store.get_preference("user-del", "k2").value == "v2"

    # Deleting non-existent key returns False
    assert temp_store.delete_preference("user-del", "k1") is False
    assert temp_store.delete_preference("user-del", "non_existent") is False


def test_list_preferences(temp_store):
    """List preferences returns all preferences for a user in deterministic order."""
    assert temp_store.list_preferences("user-list") == []

    temp_store.save_preference("user-list", "lang", "python")
    temp_store.save_preference("user-list", "editor", "vscode")
    temp_store.save_preference("other-user", "lang", "rust")

    entries = temp_store.list_preferences("user-list")
    assert len(entries) == 2
    assert [e.key for e in entries] == ["editor", "lang"]
    assert [e.value for e in entries] == ["vscode", "python"]


# ============================================================================
# 4. VALIDATION & ERROR HANDLING TESTS
# ============================================================================

def test_invalid_input_validation(temp_store):
    """18. Store raises TypeError and ValueError on invalid inputs."""
    with pytest.raises(ValueError):
        temp_store.add_message("", MessageRole.USER, "text")

    with pytest.raises(TypeError):
        temp_store.add_message("s1", "user", "text")  # string instead of MessageRole enum

    with pytest.raises(TypeError):
        temp_store.add_message("s1", MessageRole.USER, 12345)  # non-string content

    with pytest.raises(ValueError):
        temp_store.get_history("")

    with pytest.raises(ValueError):
        temp_store.save_preference("", "key", "val")

    with pytest.raises(ValueError):
        temp_store.save_preference("u1", "", "val")

    with pytest.raises(ValueError):
        temp_store.get_preference("", "key")

    with pytest.raises(ValueError):
        temp_store.get_preference("u1", "")

    with pytest.raises(ValueError):
        temp_store.delete_preference("", "key")

    with pytest.raises(ValueError):
        temp_store.delete_preference("u1", "")

    with pytest.raises(ValueError):
        temp_store.list_preferences("")


# ============================================================================
# 5. BACKWARD COMPATIBILITY SHIM TESTS
# ============================================================================

def test_history_compatibility_shim(tmp_path):
    """19. history.py public functions delegate cleanly to SQLite store."""
    test_db = tmp_path / "shim_test.db"
    store = SQLiteMemoryStore(db_path=test_db)
    history_shim.set_store(store)
    try:
        history_shim.clear_history()
        assert len(history_shim.get_history()) == 0

        history_shim.add_message("user", "Hello from shim")
        history_shim.add_message("assistant", "Hi from shim assistant")

        hist = history_shim.get_history()
        assert len(hist) == 2
        assert hist[0] == {"role": "user", "content": "Hello from shim"}
        assert hist[1] == {"role": "assistant", "content": "Hi from shim assistant"}

        # Test pop() rollback synchronization (as used in assistant.py on planning error)
        popped = hist.pop()
        assert popped == {"role": "assistant", "content": "Hi from shim assistant"}

        # Re-reading history reflects the popped item
        hist2 = history_shim.get_history()
        assert len(hist2) == 1
        assert hist2[0] == {"role": "user", "content": "Hello from shim"}

        history_shim.clear_history()
        assert len(history_shim.get_history()) == 0
    finally:
        history_shim.set_store(None)


def test_multiple_sessions_remain_independent(temp_store):
    """20. Multiple sessions interleave without cross-talk."""
    temp_store.add_message("s1", MessageRole.USER, "S1-1")
    temp_store.add_message("s2", MessageRole.USER, "S2-1")
    temp_store.add_message("s1", MessageRole.ASSISTANT, "S1-2")
    temp_store.add_message("s3", MessageRole.USER, "S3-1")
    temp_store.add_message("s2", MessageRole.ASSISTANT, "S2-2")

    assert [m.content for m in temp_store.get_history("s1")] == ["S1-1", "S1-2"]
    assert [m.content for m in temp_store.get_history("s2")] == ["S2-1", "S2-2"]
    assert [m.content for m in temp_store.get_history("s3")] == ["S3-1"]


# ============================================================================
# 6. PATH DETERMINISM & DIRECTORY CREATION TESTS
# ============================================================================

def test_database_directory_creation(tmp_path):
    """Store creates parent directories if they do not yet exist."""
    nested_path = tmp_path / "deeply" / "nested" / "dir" / "store.db"
    assert not nested_path.parent.exists()

    store = SQLiteMemoryStore(db_path=nested_path)
    assert nested_path.parent.exists()
    assert nested_path.exists()

    store.add_message("session", MessageRole.USER, "test")
    assert len(store.get_history("session")) == 1


def test_deterministic_database_path():
    """Default database path is absolute and anchored to project structure."""
    store = SQLiteMemoryStore()
    assert store.db_path.is_absolute()
    assert "storage" in store.db_path.parts
    assert store.db_path.name == "memory.db"
    assert store.db_path == DEFAULT_DB_PATH.resolve()


def test_memory_interface_abstraction():
    """MemoryServiceInterface cannot be directly instantiated."""
    with pytest.raises(TypeError):
        MemoryServiceInterface()
