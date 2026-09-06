from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from main import app
import brain.assistant as assistant_module
from memory.history import clear_history, get_history


@pytest.fixture(autouse=True)
def clean_history():
    """Ensure clean conversation history before and after each test."""
    clear_history()
    yield
    clear_history()


@pytest.fixture
def client():
    """FastAPI TestClient instance for the main application."""
    return TestClient(app)


# ============================================================================
# 1. POST /chat WITH PHASE 2 ENABLED & RESPONSE SHAPE
# ============================================================================

def test_runtime_chat_phase2_enabled_response_shape(client, monkeypatch):
    """
    1. POST /chat with Phase 2 enabled returns HTTP 200 and schema {'response': <str>}.
    """
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")

    # Mock Ollama call to be fast and deterministic while executing all Phase 2 pipeline stages
    with patch("tools.capabilities.ask_ollama", return_value="Hello there! How can I assist you?"):
        response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "response" in data
    assert isinstance(data["response"], str)
    assert data["response"] == "Hello there! How can I assist you?"


# ============================================================================
# 2. PROVE PHASE 2 WAS USED (NOT V1)
# ============================================================================

def test_runtime_chat_proves_phase2_dispatch(client, monkeypatch):
    """
    2. Proves that POST /chat dispatches to the Phase 2 pipeline and not to V1 jarvis.think().
    """
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")

    mock_jarvis = MagicMock()
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    with patch("tools.capabilities.ask_ollama", return_value="Deterministic Phase 2 response"):
        response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert response.json()["response"] == "Deterministic Phase 2 response"

    # Proof 1: Legacy V1 agent was never called
    mock_jarvis.think.assert_not_called()

    # Proof 2: Conversation history reflects Phase 2 recording
    history = get_history()
    assert len(history) >= 2
    assert history[-2]["role"] == "user"
    assert history[-2]["content"] == "hello"
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "Deterministic Phase 2 response"


# ============================================================================
# 3. EMPTY INPUT (NON-EXECUTABLE CONTROL-FLOW IN PHASE 2)
# ============================================================================

def test_runtime_chat_empty_input(client, monkeypatch):
    """
    3. POST /chat with empty input '' returns Phase 2 control-flow prompt response.
    Zero tasks executed, Ollama not contacted, V1 not called.
    """
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")

    mock_jarvis = MagicMock()
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    mock_ollama = MagicMock()
    with patch("tools.capabilities.ask_ollama", mock_ollama):
        response = client.post("/chat", json={"message": ""})

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["response"] == "Please provide an instruction or question."

    # Proof: No LLM/tool execution and V1 not called
    mock_ollama.assert_not_called()
    mock_jarvis.think.assert_not_called()


# ============================================================================
# 4. AMBIGUOUS INPUT (NON-EXECUTABLE CLARIFICATION IN PHASE 2)
# ============================================================================

def test_runtime_chat_ambiguous_input(client, monkeypatch):
    """
    4. POST /chat with ambiguous input 'open' returns Phase 2 clarification response.
    Zero tasks executed, Ollama not contacted, V1 not called.
    """
    monkeypatch.setenv("USE_PHASE2_BRAIN", "true")

    mock_jarvis = MagicMock()
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    mock_ollama = MagicMock()
    with patch("tools.capabilities.ask_ollama", mock_ollama):
        response = client.post("/chat", json={"message": "open"})

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "Please clarify your request: Incomplete query missing subject or argument." in data["response"]

    # Proof: No LLM/tool execution and V1 not called
    mock_ollama.assert_not_called()
    mock_jarvis.think.assert_not_called()


# ============================================================================
# 5. V1 MODE REMAINS SELECTABLE VIA FEATURE FLAG
# ============================================================================

def test_runtime_v1_mode_selectable(client, monkeypatch):
    """
    5. USE_PHASE2_BRAIN=false routes to legacy V1 Agent.think() and bypasses Phase 2 pipeline.
    """
    monkeypatch.setenv("USE_PHASE2_BRAIN", "false")

    mock_jarvis = MagicMock()
    mock_jarvis.think.return_value = "Response from V1 Legacy Brain"
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    mock_pipeline = MagicMock()
    monkeypatch.setattr(assistant_module, "pipeline", mock_pipeline)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert response.json()["response"] == "Response from V1 Legacy Brain"

    # Proof: V1 called, Phase 2 pipeline not called
    mock_jarvis.think.assert_called_once_with("hello")
    mock_pipeline.process.assert_not_called()


def test_runtime_v1_mode_default_when_unset(client, monkeypatch):
    """
    6. Default environment (USE_PHASE2_BRAIN unset) preserves V1 behavior.
    """
    monkeypatch.delenv("USE_PHASE2_BRAIN", raising=False)

    mock_jarvis = MagicMock()
    mock_jarvis.think.return_value = "Default V1 Response"
    monkeypatch.setattr(assistant_module, "jarvis", mock_jarvis)

    mock_pipeline = MagicMock()
    monkeypatch.setattr(assistant_module, "pipeline", mock_pipeline)

    response = client.post("/chat", json={"message": "status"})

    assert response.status_code == 200
    assert response.json()["response"] == "Default V1 Response"
    mock_jarvis.think.assert_called_once_with("status")
    mock_pipeline.process.assert_not_called()
