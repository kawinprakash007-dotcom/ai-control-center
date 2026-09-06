"""
Unit and integration tests for Phase 2.8.2 Memory <-> Phase 2 Pipeline integration.
"""

from typing import List, Optional
import pytest
from unittest.mock import MagicMock, patch

from core.interfaces.memory_interface import MemoryServiceInterface
from core.models.memory import MessageRole, ChatMessage, MemoryEntry
from core.models.request import Request
from core.models.pipeline import PipelineResult
from brain.pipeline.standard_pipeline import StandardPipeline
from brain.assistant import process_message
from brain.request_understanding import RequestUnderstandingError
import brain.assistant as assistant_module


import uuid
from datetime import datetime

class FakeMemoryService(MemoryServiceInterface):
    """Deterministic in-memory fake implementing MemoryServiceInterface."""

    def __init__(self):
        self.messages: List[ChatMessage] = []
        self.preferences: dict = {}

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[dict] = None,
        message_id: Optional[str] = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            id=message_id or f"msg-{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        self.messages.append(msg)
        return msg

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        matching = [m for m in self.messages if m.session_id == session_id]
        if limit is not None:
            return matching[-limit:]
        return matching

    def clear_session(self, session_id: str) -> None:
        self.messages = [m for m in self.messages if m.session_id != session_id]

    def save_preference(
        self,
        user_id: str,
        key: str,
        value: any,
        category: str = "general",
    ) -> MemoryEntry:
        entry = MemoryEntry(
            user_id=user_id,
            key=key,
            value=value,
            category=category,
        )
        self.preferences[(user_id, key)] = entry
        return entry

    def get_preference(
        self,
        user_id: str,
        key: str,
    ) -> Optional[MemoryEntry]:
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


def test_pipeline_retrieves_memory_using_session_id():
    """1. Pipeline retrieves memory using request.session_id."""
    fake_mem = FakeMemoryService()
    fake_mem.add_message("session-test-1", MessageRole.USER, "Prior user message")
    fake_mem.add_message("session-test-1", MessageRole.ASSISTANT, "Prior assistant response")

    pipeline = StandardPipeline(memory_service=fake_mem)

    with patch.object(pipeline.execution_engine, "execute", return_value=[]):
        res = pipeline.process({"text": "What was my last question?", "session_id": "session-test-1"})

    assert isinstance(res, PipelineResult)
    # Verify the retrieved session history had 2 messages prior to new persistence
    # And after pipeline execution, 2 new messages (user + assistant) were added
    all_msgs = fake_mem.get_history("session-test-1")
    assert len(all_msgs) == 4
    assert all_msgs[0].content == "Prior user message"
    assert all_msgs[1].content == "Prior assistant response"
    assert all_msgs[2].content == "What was my last question?"
    assert all_msgs[3].role == MessageRole.ASSISTANT


def test_memory_not_queried_for_empty_input():
    """2. Memory is not queried or written for invalid/empty input when existing semantics bypass execution."""
    fake_mem = FakeMemoryService()
    mock_mem = MagicMock(wraps=fake_mem)

    pipeline = StandardPipeline(memory_service=mock_mem)

    res = pipeline.process("")
    assert res.decision.primary_goal == "prompt_user_input"
    mock_mem.get_history.assert_not_called()
    mock_mem.add_message.assert_not_called()
    assert len(fake_mem.messages) == 0


def test_existing_conversation_context_available_to_conversational_path():
    """3. Existing conversation context is available to the conversational path (chat task parameters)."""
    fake_mem = FakeMemoryService()
    fake_mem.add_message("session-chat", MessageRole.USER, "Hello Jarvis")
    fake_mem.add_message("session-chat", MessageRole.ASSISTANT, "Greetings!")

    pipeline = StandardPipeline(memory_service=fake_mem)

    captured_plan = None

    def capture_execute(plan):
        nonlocal captured_plan
        captured_plan = plan
        return []

    with patch.object(pipeline.execution_engine, "execute", side_effect=capture_execute):
        pipeline.process({"text": "How are you?", "session_id": "session-chat"})



    assert captured_plan is not None
    assert len(captured_plan.steps) == 1
    step = captured_plan.steps[0]
    assert step.tool == "chat"
    assert step.parameters["query"] == "How are you?"
    assert step.parameters["history"] == [
        {"role": "user", "content": "Hello Jarvis"},
        {"role": "assistant", "content": "Greetings!"},
    ]


def test_current_user_message_appears_exactly_once_in_effective_llm_context():
    """4. Current user message appears exactly once in effective LLM context."""
    from tools.capabilities import chat
    from core.models.task import Task

    fake_history = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]

    task = Task(
        id=1,
        type="chat",
        action="Respond to User",
        tool="chat",
        parameters={
            "query": "New question",
            "history": fake_history,
        },
    )

    with patch("tools.capabilities.ask_ollama") as mock_ask:
        mock_ask.return_value = "Mock LLM output"
        chat(task)

        mock_ask.assert_called_once_with("New question", history=fake_history)

    # Now verify ask_ollama message building logic
    from llm.ollama_client import ask_ollama
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"message": {"content": "ok"}}
        ask_ollama("New question", history=fake_history)

        sent_payload = mock_post.call_args[1]["json"]
        sent_messages = sent_payload["messages"]

        # Expected: 1 system, 2 history, 1 user
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1] == {"role": "user", "content": "Previous question"}
        assert sent_messages[2] == {"role": "assistant", "content": "Previous answer"}
        assert sent_messages[3] == {"role": "user", "content": "New question"}
        assert len(sent_messages) == 4

        # Verify "New question" appears exactly once
        new_q_occurrences = [m for m in sent_messages if m.get("content") == "New question"]
        assert len(new_q_occurrences) == 1


def test_user_message_persisted_exactly_once():
    """5. User message is persisted exactly once."""
    fake_mem = FakeMemoryService()
    pipeline = StandardPipeline(memory_service=fake_mem)

    with patch.object(pipeline.execution_engine, "execute", return_value=[]):
        pipeline.process("Tell me a joke")

    user_msgs = [m for m in fake_mem.messages if m.role == MessageRole.USER]
    assert len(user_msgs) == 1
    assert user_msgs[0].content == "Tell me a joke"


def test_assistant_response_persisted_exactly_once():
    """6. Assistant response is persisted exactly once."""
    fake_mem = FakeMemoryService()
    pipeline = StandardPipeline(memory_service=fake_mem)

    with patch.object(pipeline.execution_engine, "execute", return_value=[]):
        res = pipeline.process("Tell me a joke")

    assistant_msgs = [m for m in fake_mem.messages if m.role == MessageRole.ASSISTANT]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].content == res.response


def test_session_a_history_not_mixed_with_session_b():
    """7. Session A history is not mixed with Session B."""
    fake_mem = FakeMemoryService()
    pipeline = StandardPipeline(memory_service=fake_mem)

    with patch.object(pipeline.execution_engine, "execute", return_value=[]):
        pipeline.process({"text": "Message A", "session_id": "session-A"})
        pipeline.process({"text": "Message B", "session_id": "session-B"})

    history_a = fake_mem.get_history("session-A")
    history_b = fake_mem.get_history("session-B")

    assert len(history_a) == 2
    assert all(m.session_id == "session-A" for m in history_a)
    assert history_a[0].content == "Message A"

    assert len(history_b) == 2
    assert all(m.session_id == "session-B" for m in history_b)
    assert history_b[0].content == "Message B"


def test_injected_fake_memory_service_without_sqlite():
    """8. Injected fake MemoryService can be used without SQLite."""
    fake_mem = FakeMemoryService()
    pipeline = StandardPipeline(memory_service=fake_mem)
    assert pipeline.memory_service is fake_mem

    with patch.object(pipeline.execution_engine, "execute", return_value=[]):
        res = pipeline.run("Testing fake memory")

    assert isinstance(res, str)
    assert len(fake_mem.messages) == 2


def test_existing_v1_bridge_behavior_remains_unchanged(monkeypatch):
    """9. Existing V1 bridge behavior remains unchanged under both USE_PHASE2_BRAIN flags."""
    # When USE_PHASE2_BRAIN is False
    monkeypatch.setenv("USE_PHASE2_BRAIN", "false")
    with patch.object(assistant_module.jarvis, "think", return_value="V1 Response") as mock_v1:
        resp = process_message("Hello V1")
        assert resp == "V1 Response"
        mock_v1.assert_called_once_with("Hello V1")

    # When USE_PHASE2_BRAIN is True
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")
    with patch.object(assistant_module.pipeline, "process") as mock_p2:
        mock_p2.return_value = MagicMock(response="Phase 2 Response")
        resp = process_message("Hello P2")
        assert resp == "Phase 2 Response"
        mock_p2.assert_called_once_with("Hello P2")


def test_request_understanding_error_reraised(monkeypatch):
    """Test RequestUnderstandingError is reraised without crashing or corrupting state."""
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")
    with patch.object(assistant_module.pipeline, "process", side_effect=RequestUnderstandingError("invalid input")):
        with pytest.raises(RequestUnderstandingError):
            process_message("Bad input")


def test_planning_failure_falls_back_to_v1(monkeypatch):
    """11. Pre-execution planning failure falls back to V1 without partial Phase 2 memory write."""
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")
    with patch.object(assistant_module.pipeline, "process", side_effect=ValueError("Unsupported capability")):
        with patch.object(assistant_module.jarvis, "think", return_value="V1 Fallback") as mock_v1:
            resp = process_message("Trigger planning error")
            assert resp == "V1 Fallback"
            mock_v1.assert_called_once_with("Trigger planning error")


def test_control_flow_prompt_user_input_persists_safely():
    """Verify prompt_user_input/clarify_request decisions persist prompt without execution."""
    fake_mem = FakeMemoryService()
    pipeline = StandardPipeline(memory_service=fake_mem)

    # clarify_request or prompt_user_input request
    res = pipeline.process({"text": "Can you do it?", "session_id": "session-clarify"})
    # Memory write happens for non-empty input
    msgs = fake_mem.get_history("session-clarify")
    assert len(msgs) == 2
    assert msgs[0].content == "Can you do it?"
    assert msgs[1].role == MessageRole.ASSISTANT
