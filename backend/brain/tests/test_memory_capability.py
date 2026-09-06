from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from core.interfaces.memory_interface import MemoryServiceInterface
from core.models.decision import CapabilityType, Decision, ExecutionMode
from core.models.memory import ChatMessage, MemoryEntry, MessageRole
from core.models.task import Task
from brain.decision_engine.standard_decision_engine import StandardDecisionEngine
from brain.request_understanding.standard_understanding import StandardRequestUnderstanding
from brain.planning.standard_planner import StandardPlanner
from brain.pipeline.standard_pipeline import StandardPipeline
from tools.memory_capability import MemoryCapability


# ============================================================================
# FAKES & TEST DOUBLES
# ============================================================================

class FakeMemoryService(MemoryServiceInterface):
    """Deterministic in-memory fake implementing MemoryServiceInterface without SQLite."""

    def __init__(self):
        self.conversations: Dict[str, List[ChatMessage]] = {}
        self.preferences: Dict[tuple[str, str], MemoryEntry] = {}
        self.save_preference_calls = 0

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
    ) -> ChatMessage:
        mid = message_id or f"msg-{len(self.conversations.get(session_id, [])) + 1}"
        msg = ChatMessage(
            id=mid,
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        self.conversations.setdefault(session_id, []).append(msg)
        return msg

    def get_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[ChatMessage]:
        msgs = self.conversations.get(session_id, [])
        if limit is not None and limit > 0:
            return msgs[-limit:]
        return list(msgs)

    def clear_session(self, session_id: str) -> None:
        if session_id in self.conversations:
            self.conversations[session_id] = []

    def save_preference(
        self, user_id: str, key: str, value: Any, category: str = "general"
    ) -> MemoryEntry:
        self.save_preference_calls += 1
        entry = MemoryEntry(
            key=key,
            value=value,
            category=category,
            user_id=user_id,
            updated_at=datetime.now(),
        )
        self.preferences[(user_id, key)] = entry
        return entry

    def get_preference(self, user_id: str, key: str) -> Optional[MemoryEntry]:
        return self.preferences.get((user_id, key))

    def delete_preference(self, user_id: str, key: str) -> bool:
        if (user_id, key) in self.preferences:
            del self.preferences[(user_id, key)]
            return True
        return False

    def list_preferences(self, user_id: str) -> List[MemoryEntry]:
        results = [entry for (uid, _), entry in self.preferences.items() if uid == user_id]
        results.sort(key=lambda e: e.key)
        return results


class FailingMemoryService(FakeMemoryService):
    """Memory service that simulates disk or storage failure on save."""

    def save_preference(
        self, user_id: str, key: str, value: Any, category: str = "general"
    ) -> MemoryEntry:
        raise IOError("Simulated storage failure during save_preference")


# ============================================================================
# 1. MEMORY CAPABILITY UNIT TESTS
# ============================================================================

def test_save_preference_through_capability():
    """1. SAVE preference stores key/value and returns deterministic success."""
    service = FakeMemoryService()
    cap = MemoryCapability(memory_service=service)

    task = Task(
        id=1,
        type="memory",
        action="Manage Memory",
        tool="memory",
        parameters={
            "action": "save",
            "key": "favorite_language",
            "value": "Python",
            "user_id": "default_user",
        },
    )

    response = cap(task)
    assert response == "Got it — I'll remember that your favorite language is Python."
    stored = service.get_preference("default_user", "favorite_language")
    assert stored is not None
    assert stored.value == "Python"


def test_read_existing_preference():
    """2. READ returns the saved preference value deterministically."""
    service = FakeMemoryService()
    service.save_preference("default_user", "favorite_language", "Python")
    cap = MemoryCapability(memory_service=service)

    task = Task(
        id=1,
        type="memory",
        action="Manage Memory",
        tool="memory",
        parameters={
            "action": "read",
            "key": "favorite_language",
            "user_id": "default_user",
        },
    )

    response = cap(task)
    assert response == "Your favorite language is Python."


def test_read_missing_preference():
    """3. READ returns not-found message when preference key does not exist."""
    service = FakeMemoryService()
    cap = MemoryCapability(memory_service=service)

    task = Task(
        id=1,
        type="memory",
        action="Manage Memory",
        tool="memory",
        parameters={
            "action": "read",
            "key": "car_model",
            "user_id": "default_user",
        },
    )

    response = cap(task)
    assert response == "I don't have anything remembered for your car model."


def test_read_all_preferences():
    """4. READ 'all' returns all saved preferences or empty message."""
    service = FakeMemoryService()
    cap = MemoryCapability(memory_service=service)

    # Empty store
    task_empty = Task(
        id=1,
        type="memory",
        action="Manage Memory",
        tool="memory",
        parameters={"action": "read", "key": "all", "user_id": "default_user"},
    )
    assert cap(task_empty) == "I don't have any memories stored about you yet."

    # Populated store
    service.save_preference("default_user", "favorite_language", "Python")
    service.save_preference("default_user", "preferred_editor", "VS Code")

    response = cap(task_empty)
    assert "I currently remember:" in response
    assert "- favorite language: Python" in response
    assert "- preferred editor: VS Code" in response


def test_forget_existing_preference():
    """5. FORGET deletes target preference and returns success."""
    service = FakeMemoryService()
    service.save_preference("default_user", "favorite_language", "Python")
    cap = MemoryCapability(memory_service=service)

    task = Task(
        id=1,
        type="memory",
        action="Manage Memory",
        tool="memory",
        parameters={
            "action": "forget",
            "key": "favorite_language",
            "user_id": "default_user",
        },
    )

    response = cap(task)
    assert response == "Okay — I've forgotten that preference."
    assert service.get_preference("default_user", "favorite_language") is None


def test_forget_missing_preference():
    """6. FORGET returns informative message when key was not found."""
    service = FakeMemoryService()
    cap = MemoryCapability(memory_service=service)

    task = Task(
        id=1,
        type="memory",
        action="Manage Memory",
        tool="memory",
        parameters={
            "action": "forget",
            "key": "favorite_language",
            "user_id": "default_user",
        },
    )

    response = cap(task)
    assert response == "I couldn't find any memory for your favorite language to forget."


def test_forget_one_does_not_delete_another():
    """7. FORGET deletes only targeted preference and leaves others intact."""
    service = FakeMemoryService()
    service.save_preference("default_user", "favorite_language", "Python")
    service.save_preference("default_user", "editor", "VS Code")
    cap = MemoryCapability(memory_service=service)

    task = Task(
        id=1,
        type="memory",
        action="Manage Memory",
        tool="memory",
        parameters={"action": "forget", "key": "favorite_language"},
    )
    cap(task)

    assert service.get_preference("default_user", "favorite_language") is None
    assert service.get_preference("default_user", "editor") is not None
    assert service.get_preference("default_user", "editor").value == "VS Code"


# ============================================================================
# 2. DETERMINISTIC RECOGNITION TESTS
# ============================================================================

def test_explicit_save_recognized_as_memory():
    """8. Explicit SAVE directives are classified as CapabilityType.MEMORY."""
    engine = StandardDecisionEngine()
    understand = StandardRequestUnderstanding()

    examples = [
        ("remember that my favorite language is Python", "favorite_language", "Python"),
        ("remember my favorite language is Python", "favorite_language", "Python"),
        ("save that my project is called ATLAS", "project", "ATLAS"),
        ("store preference that my editor is VS Code", "editor", "VS Code"),
        ("remember my preferred editor is VS Code", "preferred_editor", "VS Code"),
    ]

    for phrase, expected_key, expected_val in examples:
        req = understand.understand(phrase)
        dec = engine.decide(req)
        assert CapabilityType.MEMORY in dec.required_capabilities, f"Failed on: {phrase}"
        assert dec.primary_goal == "manage_memory"
        assert dec.routing_hints.get("action") == "save"
        assert dec.routing_hints.get("key") == expected_key
        assert dec.routing_hints.get("value") == expected_val


def test_explicit_read_recognized_as_memory():
    """9. Explicit READ queries are classified as CapabilityType.MEMORY."""
    engine = StandardDecisionEngine()
    understand = StandardRequestUnderstanding()

    examples = [
        ("what do you remember about me", "all"),
        ("what do you know about me", "all"),
        ("what is my favorite language", "favorite_language"),
        ("what did I tell you my favorite language was", "favorite_language"),
        ("recall my favorite language", "favorite_language"),
    ]

    for phrase, expected_key in examples:
        req = understand.understand(phrase)
        dec = engine.decide(req)
        assert CapabilityType.MEMORY in dec.required_capabilities, f"Failed on: {phrase}"
        assert dec.primary_goal == "manage_memory"
        assert dec.routing_hints.get("action") == "read"
        assert dec.routing_hints.get("key") == expected_key


def test_explicit_forget_recognized_as_memory():
    """10. Explicit FORGET directives are classified as CapabilityType.MEMORY."""
    engine = StandardDecisionEngine()
    understand = StandardRequestUnderstanding()

    examples = [
        ("forget my favorite language", "favorite_language"),
        ("forget that my favorite language is Python", "favorite_language"),
        ("delete my saved preference for editor", "editor"),
        ("forget that my project is called ATLAS", "project"),
    ]

    for phrase, expected_key in examples:
        req = understand.understand(phrase)
        dec = engine.decide(req)
        assert CapabilityType.MEMORY in dec.required_capabilities, f"Failed on: {phrase}"
        assert dec.primary_goal == "manage_memory"
        assert dec.routing_hints.get("action") == "forget"
        assert dec.routing_hints.get("key") == expected_key


def test_generic_what_is_recall_precedes_knowledge():
    """11. 'What is my...' routes to MEMORY; 'What is...' routes to KNOWLEDGE."""
    engine = StandardDecisionEngine()
    understand = StandardRequestUnderstanding()

    # Personal recall goes to MEMORY
    req_mem = understand.understand("what is my favorite language")
    dec_mem = engine.decide(req_mem)
    assert dec_mem.required_capabilities == [CapabilityType.MEMORY]
    assert dec_mem.primary_goal == "manage_memory"

    # Generic query goes to KNOWLEDGE
    req_know = understand.understand("what is Linux")
    dec_know = engine.decide(req_know)
    assert dec_know.required_capabilities == [CapabilityType.KNOWLEDGE]
    assert dec_know.primary_goal == "retrieve_knowledge"


def test_ordinary_statements_remain_chat():
    """12. Normal conversational statements without directive verbs stay CHAT."""
    engine = StandardDecisionEngine()
    understand = StandardRequestUnderstanding()

    chat_phrases = [
        "Python is my favorite language.",
        "I really like Python.",
        "My project is called ATLAS.",
        "What do you think about Python?",
    ]

    for phrase in chat_phrases:
        req = understand.understand(phrase)
        dec = engine.decide(req)
        assert CapabilityType.MEMORY not in dec.required_capabilities, f"Incorrectly marked memory: {phrase}"
        assert dec.required_capabilities == [CapabilityType.CHAT]
        assert dec.primary_goal == "answer_chat"


def test_ambiguous_memory_command_does_not_mutate_storage():
    """13. Ambiguous or incomplete memory operations do not alter storage."""
    service = FakeMemoryService()
    cap = MemoryCapability(memory_service=service)

    # Missing key
    task_no_key = Task(
        id=1,
        type="memory",
        action="Manage Memory",
        tool="memory",
        parameters={"action": "save", "key": "", "value": "Python"},
    )
    res_no_key = cap(task_no_key)
    assert "Cannot save memory" in res_no_key
    assert len(service.preferences) == 0

    # Missing value
    task_no_val = Task(
        id=2,
        type="memory",
        action="Manage Memory",
        tool="memory",
        parameters={"action": "save", "key": "favorite_language", "value": ""},
    )
    res_no_val = cap(task_no_val)
    assert "Cannot save memory" in res_no_val
    assert len(service.preferences) == 0


def test_injected_fake_memory_service_works_without_sqlite():
    """14. FakeMemoryService completely isolates capability from SQLite."""
    service = FakeMemoryService()
    cap = MemoryCapability(memory_service=service)
    assert cap.memory_service is service


def test_memory_capability_does_not_directly_import_sqlite_store():
    """15. MemoryCapability module does not import SQLiteMemoryStore or sqlite3."""
    import inspect
    import tools.memory_capability as mc_mod

    source = inspect.getsource(mc_mod)
    assert "sqlite3" not in source.lower(), "Direct sqlite3 import found in memory_capability"
    assert "from memory.sqlite_store import SQLiteMemoryStore" not in source, (
        "Direct SQLiteMemoryStore import found in memory_capability"
    )


# ============================================================================
# 3. PLANNER & PIPELINE INTEGRATION TESTS
# ============================================================================

def test_planner_passes_memory_parameters():
    """16. StandardPlanner propagates memory parameters to Task."""
    planner = StandardPlanner()
    decision = Decision(
        request_id="req-1",
        primary_goal="manage_memory",
        required_capabilities=[CapabilityType.MEMORY],
        execution_mode=ExecutionMode.SINGLE_STEP,
        confidence=0.95,
        reasoning="Memory directive",
        routing_hints={
            "action": "save",
            "key": "favorite_language",
            "value": "Python",
            "user_id": "default_user",
        },
    )

    plan = planner.plan(decision)
    assert len(plan.steps) == 1
    task = plan.steps[0]
    assert task.type == "memory"
    assert task.parameters["action"] == "save"
    assert task.parameters["key"] == "favorite_language"
    assert task.parameters["value"] == "Python"
    assert task.parameters["user_id"] == "default_user"


def test_pipeline_executes_memory_capability_normal_path():
    """17. Pipeline runs SAVE, READ, FORGET through full Phase 2 lifecycle."""
    fake_service = FakeMemoryService()
    pipeline = StandardPipeline(memory_service=fake_service)

    # Turn 1: SAVE
    res1 = pipeline.process("Remember that my favorite language is Python.")
    assert "Got it — I'll remember that your favorite language is Python." in res1.response
    assert res1.verification.verified is True
    assert fake_service.get_preference("default_user", "favorite_language") is not None

    # Turn 2: READ
    res2 = pipeline.process("What is my favorite language?")
    assert "Your favorite language is Python." in res2.response
    assert res2.verification.verified is True

    # Turn 3: FORGET
    res3 = pipeline.process("Forget my favorite language.")
    assert "Okay — I've forgotten that preference." in res3.response
    assert res3.verification.verified is True
    assert fake_service.get_preference("default_user", "favorite_language") is None


def test_verification_succeeds_for_memory_operation():
    """18. Verifier marks successful memory operations verified."""
    fake_service = FakeMemoryService()
    pipeline = StandardPipeline(memory_service=fake_service)

    result = pipeline.process("Remember that my favorite language is Python.")
    assert result.verification.verified is True
    assert result.verification.status == "verified"


def test_storage_failure_does_not_create_false_success():
    """19. Storage failure reports error and never produces false confirmation."""
    failing_service = FailingMemoryService()
    pipeline = StandardPipeline(memory_service=failing_service)

    result = pipeline.process("Remember that my favorite language is Python.")
    assert result.verification.verified is False
    assert result.verification.status == "failed"
    assert "Execution failed" in result.response
    assert "Got it — I'll remember" not in result.response


def test_explicit_preference_persistence_occurs_exactly_once():
    """20. Explicit preference persistence is executed exactly once per turn."""
    fake_service = FakeMemoryService()
    pipeline = StandardPipeline(memory_service=fake_service)

    pipeline.process("Remember that my favorite language is Python.")
    assert fake_service.save_preference_calls == 1
    assert len(fake_service.preferences) == 1


def test_conversation_history_remains_separate_from_preferences():
    """21. Session conversation history is distinct from explicit preferences."""
    fake_service = FakeMemoryService()
    pipeline = StandardPipeline(memory_service=fake_service)

    pipeline.process("Remember that my favorite language is Python.")

    # 1. Preferences has only the explicit key-value
    assert len(fake_service.preferences) == 1
    pref = fake_service.get_preference("default_user", "favorite_language")
    assert pref is not None and pref.value == "Python"

    # 2. History has the conversational turns
    history = fake_service.get_history("default")
    assert len(history) == 2
    assert history[0].role == MessageRole.USER
    assert history[0].content == "Remember that my favorite language is Python."
    assert history[1].role == MessageRole.ASSISTANT
    assert "Got it — I'll remember" in history[1].content
