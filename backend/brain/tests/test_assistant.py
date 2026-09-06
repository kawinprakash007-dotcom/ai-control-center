import pytest
from unittest.mock import MagicMock

import brain.assistant as assistant_module
from brain.assistant import process_message, is_phase2_enabled
from core.models.pipeline import PipelineResult
from core.models.result import Result
from core.models.verification import VerificationResult
from core.models.plan import Plan
from core.models.task import Task
from core.models.decision import Decision, CapabilityType, ExecutionMode
from core.models.request import Request
from brain.execution import StandardExecutionEngine
from brain.pipeline import StandardPipeline
from memory.history import clear_history, get_history, add_message


class StubRouter:
    def __init__(self, default_result=None, responses=None):
        self.default_result = default_result or Result(
            success=True, message="Success", output="Stub Router Output"
        )
        self.responses = responses or {}
        self.routed_tasks = []

    def route(self, task):
        self.routed_tasks.append(task)
        return self.responses.get(task.id, self.default_result)


@pytest.fixture(autouse=True)
def clean_history():
    """Ensure clean conversation history before and after each test."""
    clear_history()
    yield
    clear_history()


# ============================================================================
# 1. V1 MODE TESTS
# ============================================================================

def test_v1_mode_calls_agent_think(monkeypatch):
    """1. USE_PHASE2_BRAIN=False dispatches directly to jarvis.think()."""
    monkeypatch.delenv("USE_PHASE2_BRAIN", raising=False)
    assert not is_phase2_enabled()

    mock_jarvis = MagicMock()
    mock_jarvis.think.return_value = "V1 Agent Response"
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    mock_pipeline = MagicMock()
    monkeypatch.setattr(assistant_module, "pipeline", mock_pipeline)

    res = process_message("hello")

    assert res == "V1 Agent Response"
    mock_jarvis.think.assert_called_once_with("hello")
    mock_pipeline.process.assert_not_called()


# ============================================================================
# 2. PHASE 2 MODE TESTS
# ============================================================================

def test_phase2_mode_calls_pipeline(monkeypatch):
    """2. USE_PHASE2_BRAIN=True dispatches to Phase 2 pipeline and not Agent.think()."""
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")
    assert is_phase2_enabled()

    mock_jarvis = MagicMock()
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    router = StubRouter(default_result=Result(success=True, message="OK", output="Hello from Phase 2"))
    stub_pipeline = StandardPipeline(execution_engine=StandardExecutionEngine(router=router))
    monkeypatch.setattr(assistant_module, "pipeline", stub_pipeline)

    res = process_message("hello")

    assert res == "Hello from Phase 2"
    mock_jarvis.think.assert_not_called()


# ============================================================================
# 3. KNOWLEDGE ROUTING IN PHASE 2
# ============================================================================

def test_phase2_knowledge_routing(monkeypatch):
    """3. Knowledge query traverses Phase 2 pipeline and returns document excerpt."""
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")

    mock_jarvis = MagicMock()
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    knowledge_result = Result(
        success=True,
        message="Retrieved",
        output="VFS provides an abstract interface to file systems.",
    )
    router = StubRouter(default_result=knowledge_result)
    stub_pipeline = StandardPipeline(execution_engine=StandardExecutionEngine(router=router))
    monkeypatch.setattr(assistant_module, "pipeline", stub_pipeline)

    res = process_message("explain linux kernel vfs architecture")

    assert res == "VFS provides an abstract interface to file systems."
    assert len(router.routed_tasks) == 1
    assert router.routed_tasks[0].type == "knowledge"
    assert router.routed_tasks[0].tool == "knowledge"
    mock_jarvis.think.assert_not_called()


# ============================================================================
# 4. TOOL ROUTING IN PHASE 2
# ============================================================================

def test_phase2_tool_routing(monkeypatch):
    """4. Tool request traverses Phase 2 pipeline using stub router without launching real apps."""
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")

    mock_jarvis = MagicMock()
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    tool_result = Result(success=True, message="Notepad launched.", output=None)
    router = StubRouter(default_result=tool_result)
    stub_pipeline = StandardPipeline(execution_engine=StandardExecutionEngine(router=router))
    monkeypatch.setattr(assistant_module, "pipeline", stub_pipeline)

    res = process_message("open notepad")

    assert res == "Notepad launched."
    assert len(router.routed_tasks) == 1
    assert router.routed_tasks[0].type == "tool"
    assert router.routed_tasks[0].tool == "notepad"
    mock_jarvis.think.assert_not_called()


# ============================================================================
# 5. EMPTY INPUT (NON-EXECUTABLE CONTROL-FLOW)
# ============================================================================

def test_phase2_empty_input(monkeypatch):
    """5. Empty input returns prompt instruction; no tool/Ollama execution, V1 not called."""
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")

    mock_jarvis = MagicMock()
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    router = StubRouter()
    stub_pipeline = StandardPipeline(execution_engine=StandardExecutionEngine(router=router))
    monkeypatch.setattr(assistant_module, "pipeline", stub_pipeline)

    res = process_message("")

    assert res == "Please provide an instruction or question."
    assert len(router.routed_tasks) == 0
    mock_jarvis.think.assert_not_called()


# ============================================================================
# 6. AMBIGUOUS INPUT (NON-EXECUTABLE CONTROL-FLOW)
# ============================================================================

def test_phase2_ambiguous_input(monkeypatch):
    """6. Ambiguous command 'open' returns clarification; no tool execution, V1 not called."""
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")

    mock_jarvis = MagicMock()
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    router = StubRouter()
    stub_pipeline = StandardPipeline(execution_engine=StandardExecutionEngine(router=router))
    monkeypatch.setattr(assistant_module, "pipeline", stub_pipeline)

    res = process_message("open")

    assert "Please clarify your request" in res
    assert len(router.routed_tasks) == 0
    mock_jarvis.think.assert_not_called()


# ============================================================================
# 7. PRE-EXECUTION UNSUPPORTED CAPABILITY FALLBACK TO V1
# ============================================================================

def test_pre_execution_unsupported_fallback_to_v1(monkeypatch):
    """7. Unsupported capability rejected during planning safely falls back to V1 before execution."""
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")

    mock_jarvis = MagicMock()
    mock_jarvis.think.return_value = "V1 Fallback Vision Response"
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    router = StubRouter()
    # StandardPipeline with default StandardPlanner will raise ValueError on vision requests
    stub_pipeline = StandardPipeline(execution_engine=StandardExecutionEngine(router=router))
    monkeypatch.setattr(assistant_module, "pipeline", stub_pipeline)

    # "screenshot" triggers CapabilityType.VISION in StandardDecisionEngine, rejected by StandardPlanner
    res = process_message("take a screenshot of the screen")

    assert res == "V1 Fallback Vision Response"
    mock_jarvis.think.assert_called_once_with("take a screenshot of the screen")
    # Crucial safety check: Phase 2 execution was never started
    assert len(router.routed_tasks) == 0


# ============================================================================
# 8. POST-EXECUTION FAILURE (NO FALLBACK TO V1)
# ============================================================================

def test_post_execution_failure_does_not_call_v1(monkeypatch):
    """8. Execution failure occurring after task execution begins must NOT fall back to V1."""
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")

    mock_jarvis = MagicMock()
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    failing_result = Result(success=False, message="Device offline error", output=None)
    router = StubRouter(default_result=failing_result)
    stub_pipeline = StandardPipeline(execution_engine=StandardExecutionEngine(router=router))
    monkeypatch.setattr(assistant_module, "pipeline", stub_pipeline)

    res = process_message("open calculator")

    # Phase 2 response handles error; V1 is strictly NOT called
    assert "Execution failed on task 1" in res
    assert "Device offline error" in res
    mock_jarvis.think.assert_not_called()


# ============================================================================
# 9. HISTORY BEHAVIOR
# ============================================================================

def test_history_behavior_in_phase2(monkeypatch):
    """9. Phase 2 records user and assistant messages exactly once in history."""
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")

    router = StubRouter(default_result=Result(success=True, message="OK", output="Assistant reply"))
    stub_pipeline = StandardPipeline(execution_engine=StandardExecutionEngine(router=router))
    monkeypatch.setattr(assistant_module, "pipeline", stub_pipeline)

    res = process_message("User query")

    assert res == "Assistant reply"
    history = get_history()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "User query"}
    assert history[1] == {"role": "assistant", "content": "Assistant reply"}


def test_history_behavior_on_fallback(monkeypatch):
    """9. On pre-execution fallback to V1, user message is not duplicated in history."""
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")

    def mock_think(msg):
        # Emulate V1 Agent.think() recording history
        add_message("user", msg)
        add_message("assistant", "V1 Response")
        return "V1 Response"

    mock_jarvis = MagicMock()
    mock_jarvis.think.side_effect = mock_think
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    router = StubRouter()
    stub_pipeline = StandardPipeline(execution_engine=StandardExecutionEngine(router=router))
    monkeypatch.setattr(assistant_module, "pipeline", stub_pipeline)

    res = process_message("take a screenshot")

    assert res == "V1 Response"
    history = get_history()
    # History must contain exactly 1 user message and 1 assistant message, not 2 user messages
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "take a screenshot"}
    assert history[1] == {"role": "assistant", "content": "V1 Response"}


# ============================================================================
# 10. FEATURE FLAG BEHAVIOR
# ============================================================================

def test_feature_flag_variants(monkeypatch):
    """10. Feature flag parses truthy and falsy values deterministically."""
    for truthy_val in ("true", "True", "TRUE", "1", "yes", "YES", "on"):
        monkeypatch.setenv("USE_PHASE2_BRAIN", truthy_val)
        assert is_phase2_enabled() is True

    for falsy_val in ("false", "False", "0", "no", "NO", "off", "", "random"):
        monkeypatch.setenv("USE_PHASE2_BRAIN", falsy_val)
        assert is_phase2_enabled() is False

    monkeypatch.delenv("USE_PHASE2_BRAIN", raising=False)
    assert is_phase2_enabled() is False
