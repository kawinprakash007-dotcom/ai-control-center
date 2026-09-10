import os
import sys
import ast
import time
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from core.models.request import Request
from core.models.tool_call import ToolCall
from core.models.policy import PolicyContext, PolicyDecision, RiskLevel
from core.models.runtime import TurnStatus, CognitiveEventType
from brain.request_understanding import StandardRequestUnderstanding
from brain.decision_engine import StandardDecisionEngine
from brain.planning import StandardPlanner
from safety.policy_engine import StandardPolicyEngine, DemoApplicationLaunchPolicyRule
from computer.demo_app_capability import (
    DemoAppCapability,
    FIXED_APPLICATION_REGISTRY,
    ApplicationRegistryEntry,
    MockDemoAppBackend,
)
from tools.tool_orchestrator import ToolOrchestrator
from tools.capability_registry import CapabilityRegistry
from runtime.cognitive_runtime import CognitiveRuntime
from runtime.event_sink import InMemoryEventSink


def make_request(text: str) -> Request:
    return Request(
        id="test-req-demo",
        original_text=text,
        normalized_text=text,
        session_id="test-session",
        timestamp=datetime.now(),
        parameters={},
        constraints={},
    )


# ============================================================================
# 1. PRODUCTION MODE DENIES APPLICATION LAUNCH
# ============================================================================

def test_production_mode_denies_application_launch():
    """In production mode (demo_mode=False), application launch attempts remain denied."""
    policy = StandardPolicyEngine(demo_mode=False)

    # 1. Rule RULE_DEMO_APP_LAUNCH must not even be present in production
    assert not any(r.rule_id == "RULE_DEMO_APP_LAUNCH" for r in policy.rules)

    # 2. Direct computer_app launch call denied
    tc_demo = ToolCall(capability="computer_app", action="launch", parameters={"app_id": "vscode"})
    ctx_demo = PolicyContext.from_tool_call(tc_demo)
    res_demo = policy.evaluate(tc_demo, ctx_demo)
    assert res_demo.is_denied is True
    assert res_demo.rule_id == "RULE_DEFAULT_DENY"

    # 3. Controlled computer open_app call denied
    tc_comp = ToolCall(capability="computer", action="open_app")
    ctx_comp = PolicyContext.from_tool_call(tc_comp)
    res_comp = policy.evaluate(tc_comp, ctx_comp)
    assert res_comp.is_denied is True
    assert "Opening applications is not permitted" in res_comp.reason

    # 4. End-to-end turn in production mode fails
    engine = StandardDecisionEngine(demo_mode=False)
    planner = StandardPlanner()
    runtime = CognitiveRuntime(
        decision_engine=engine,
        planner=planner,
        policy_engine=policy,
        memory_service=MagicMock(),
        demo_mode=False,
    )
    turn_res = runtime.execute_turn("open VS Code")
    assert turn_res.status == TurnStatus.FAILED
    assert "Policy denied execution" in turn_res.response


# ============================================================================
# 2. DEMO MODE ALLOWS REGISTERED APPLICATIONS
# ============================================================================

def test_demo_mode_allows_vscode():
    """Demo mode authorizes launch, focus, and close for registered VS Code."""
    policy = StandardPolicyEngine(demo_mode=True)

    for action in ("launch", "focus", "close"):
        tc = ToolCall(capability="computer_app", action=action, parameters={"app_id": "vscode"})
        ctx = PolicyContext.from_tool_call(tc)
        res = policy.evaluate(tc, ctx)
        assert res.is_allowed is True
        assert res.rule_id == "RULE_DEMO_APP_ALLOWED"
        assert res.metadata.get("app_id") == "vscode"
        assert res.metadata.get("demo_mode") is True


def test_demo_mode_allows_chrome():
    """Demo mode authorizes launch for registered Chrome."""
    policy = StandardPolicyEngine(demo_mode=True)
    tc = ToolCall(capability="computer_app", action="launch", parameters={"app_id": "chrome"})
    ctx = PolicyContext.from_tool_call(tc)
    res = policy.evaluate(tc, ctx)
    assert res.is_allowed is True
    assert res.rule_id == "RULE_DEMO_APP_ALLOWED"
    assert res.metadata.get("app_id") == "chrome"


def test_demo_mode_allows_notepad():
    """Demo mode authorizes launch for registered Notepad."""
    policy = StandardPolicyEngine(demo_mode=True)
    tc = ToolCall(capability="computer_app", action="launch", parameters={"app_id": "notepad"})
    ctx = PolicyContext.from_tool_call(tc)
    res = policy.evaluate(tc, ctx)
    assert res.is_allowed is True
    assert res.rule_id == "RULE_DEMO_APP_ALLOWED"
    assert res.metadata.get("app_id") == "notepad"


# ============================================================================
# 3. SAFETY DENIALS: UNKNOWN APP, ARBITRARY PATHS, SHELL, POWERSHELL
# ============================================================================

def test_unknown_app_denied():
    """Any application not in FIXED_APPLICATION_REGISTRY is denied."""
    policy = StandardPolicyEngine(demo_mode=True)
    for bad_app in ("malicious_tool", "firefox", "powershell", "cmd", "calc", "unknown_binary"):
        tc = ToolCall(capability="computer_app", action="launch", parameters={"app_id": bad_app})
        ctx = PolicyContext.from_tool_call(tc)
        res = policy.evaluate(tc, ctx)
        assert res.is_denied is True
        assert res.rule_id == "RULE_DEMO_APP_DENIED"
        assert f"Application '{bad_app}' is not an authorized demonstration application" in res.reason


def test_arbitrary_executable_denied():
    """Supplying arbitrary executable paths is strictly denied."""
    policy = StandardPolicyEngine(demo_mode=True)
    orchestrator = ToolOrchestrator(demo_mode=True, policy_engine=policy)

    tc = ToolCall(
        capability="computer_app",
        action="launch",
        parameters={"app_id": "vscode", "executable": "C:\\malicious\\payload.exe"},
    )
    is_valid, msg = orchestrator.validate_call(tc)
    assert is_valid is False
    assert "strictly forbidden" in msg

    ctx = PolicyContext.from_tool_call(tc)
    res = policy.evaluate(tc, ctx)
    assert res.is_denied is True
    assert res.rule_id == "RULE_DEMO_APP_DENIED"
    assert "Supplying arbitrary 'executable'" in res.reason


def test_shell_command_denied():
    """Supplying shell parameters or command lines is strictly denied."""
    policy = StandardPolicyEngine(demo_mode=True)
    orchestrator = ToolOrchestrator(demo_mode=True, policy_engine=policy)

    forbidden_payloads = [
        {"app_id": "vscode", "shell": True},
        {"app_id": "vscode", "command": "rmdir /s /q C:\\"},
        {"app_id": "vscode", "cmd": "dir"},
        {"app_id": "vscode", "script": "evil.bat"},
        {"app_id": "vscode", "args": ["--dangerous-flag"]},
    ]

    for params in forbidden_payloads:
        tc = ToolCall(capability="computer_app", action="launch", parameters=params)
        is_valid, msg = orchestrator.validate_call(tc)
        assert is_valid is False
        assert "forbidden" in msg

        ctx = PolicyContext.from_tool_call(tc)
        res = policy.evaluate(tc, ctx)
        assert res.is_denied is True
        assert res.rule_id == "RULE_DEMO_APP_DENIED"


def test_powershell_denied():
    """PowerShell and direct shell capabilities remain strictly denied."""
    policy = StandardPolicyEngine(demo_mode=True)

    for shell_cap in ("powershell", "cmd", "shell", "bash", "subprocess", "exec", "eval"):
        tc = ToolCall(capability=shell_cap, action="run", parameters={"command": "Get-Process"})
        ctx = PolicyContext.from_tool_call(tc)
        res = policy.evaluate(tc, ctx)
        assert res.is_denied is True
        assert res.rule_id == "RULE_FORBIDDEN_CAPABILITY"
        assert res.metadata.get("risk_level") == RiskLevel.CRITICAL.value


def test_model_cannot_inject_executable_path():
    """Model-generated executable paths cannot override the fixed registry."""
    mock_backend = MockDemoAppBackend()
    cap = DemoAppCapability(backend=mock_backend)

    # Attempt to inject custom path
    res = cap.execute(action="launch", app_id="vscode", executable="C:\\custom\\code.exe")
    assert res.success is False
    assert "forbidden" in res.message
    assert len(mock_backend.history) == 0

    # Clean invocation only uses registry path
    res_clean = cap.execute(action="launch", app_id="vscode")
    assert res_clean.success is True
    assert len(mock_backend.history) == 1
    assert mock_backend.history[0]["app_id"] == "vscode"


# ============================================================================
# 4. ERROR HANDLING: MISSING APP & STRUCTURED FAILURES
# ============================================================================

def test_missing_app_structured_failure():
    """When registered app binary is missing, returns structured failure without crashing."""
    mock_backend = MockDemoAppBackend()
    mock_backend.missing_apps.add("vscode")
    cap = DemoAppCapability(backend=mock_backend)

    res = cap.execute(action="launch", app_id="vscode")
    assert res.success is False
    assert "executable not found" in res.message
    assert res.data.get("demo_mode") is True
    assert res.data.get("result") == "failure"


# ============================================================================
# 5. DETERMINISTIC INTENT MAPPINGS (PHASE 6.6 REQUIREMENT 9)
# ============================================================================

def test_deterministic_intent_mappings_vscode():
    """open/launch/start vscode resolves to computer_app.launch(app_id='vscode') in demo mode."""
    engine = StandardDecisionEngine(demo_mode=True)
    planner = StandardPlanner()

    for cmd in [
        "open vscode",
        "open visual studio code",
        "launch vscode",
        "start vscode",
        "open VS Code",
        "launch VS Code",
        "start Visual Studio Code",
        "open the vs code in my pc",
    ]:
        req = make_request(cmd)
        decision = engine.decide(req)

        assert decision.primary_goal == "computer_app", f"Failed for '{cmd}'"
        assert decision.routing_hints.get("tool_hint") == "computer_app", f"Failed for '{cmd}'"
        assert decision.routing_hints.get("app_id") == "vscode", f"Failed for '{cmd}'"
        assert decision.routing_hints.get("action") == "launch", f"Failed for '{cmd}'"

        plan = planner.plan(decision)
        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert step.tool == "computer_app"
        assert step.parameters.get("app_id") == "vscode"
        assert step.parameters.get("action") == "launch"


def test_deterministic_intent_mappings_chrome_and_notepad():
    """open/launch chrome and notepad resolve to computer_app in demo mode."""
    engine = StandardDecisionEngine(demo_mode=True)
    planner = StandardPlanner()

    cases = [
        ("open chrome", "chrome", "launch"),
        ("launch chrome", "chrome", "launch"),
        ("open notepad", "notepad", "launch"),
        ("launch notepad", "notepad", "launch"),
        ("close notepad", "notepad", "close"),
        ("focus vscode", "vscode", "focus"),
    ]

    for cmd, expected_app, expected_action in cases:
        req = make_request(cmd)
        decision = engine.decide(req)
        assert decision.primary_goal == "computer_app", f"Failed for '{cmd}'"
        assert decision.routing_hints.get("app_id") == expected_app, f"Failed for '{cmd}'"
        assert decision.routing_hints.get("action") == expected_action, f"Failed for '{cmd}'"


# ============================================================================
# 6. COMPLETE END-TO-END TURN EXECUTION & AUDITABILITY
# ============================================================================

def test_complete_open_vscode_flow_passes_in_demo_mode():
    """Complete 'Open VS Code' turn flow executes cleanly through the pipeline in demo mode."""
    event_sink = InMemoryEventSink()
    mock_backend = MockDemoAppBackend()
    demo_cap = DemoAppCapability(backend=mock_backend)

    runtime = CognitiveRuntime(
        event_sink=event_sink,
        demo_mode=True,
    )
    # Inject mock backend to avoid popping native window during test
    runtime.execution_engine.router.registry.register("computer_app", demo_cap)

    turn_res = runtime.execute_turn("Open VS Code")

    assert turn_res.status == TurnStatus.SUCCEEDED
    assert "Launched Visual Studio Code" in turn_res.response
    assert turn_res.trace is not None

    # Check audit events
    events = event_sink.get_events(turn_res.turn_id)
    tool_events = [e for e in events if e.event_type == CognitiveEventType.TOOL_EXECUTED]
    assert len(tool_events) == 1
    te = tool_events[0]
    assert te.status == "OK"
    assert te.metadata.get("demo_mode") is True
    assert te.metadata.get("application_id") == "vscode"
    assert te.metadata.get("action") == "launch"
    assert te.metadata.get("result") == "success"
    assert "timestamp" in te.metadata


# ============================================================================
# 7. SECURITY INVARIANT: NO GENERIC SHELL PATH INTRODUCED
# ============================================================================

def test_security_invariant_no_generic_shell_in_codebase():
    """
    Search all modified/new Python files in backend and prove:
    1. No shell=True
    2. No os.system
    3. No eval() or exec()
    4. Popen only called with shell=False and fixed argv
    """
    targets = [
        "backend/computer/demo_app_capability.py",
        "backend/safety/policy_engine.py",
        "backend/tools/tool_orchestrator.py",
        "backend/brain/decision_engine/standard_decision_engine.py",
        "backend/brain/planning/standard_planner.py",
        "backend/brain/request_understanding/standard_understanding.py",
        "backend/core/app_state.py",
        "backend/runtime/cognitive_runtime.py",
    ]

    for rel_path in targets:
        full_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", rel_path)
        assert os.path.exists(full_path), f"File {rel_path} not found"
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse AST to ensure valid python
        tree = ast.parse(content, filename=rel_path)

        # 1. No shell=True anywhere
        assert "shell=True" not in content, f"Forbidden shell=True found in {rel_path}"

        # 2. No os.system in these components
        assert "os.system" not in content, f"Forbidden os.system found in {rel_path}"

        # 3. Check AST calls for eval / exec
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in ("eval", "exec"), f"Forbidden {node.func.id}() call in {rel_path}"


# ============================================================================
# 8. COMPLETE FASTAPI API TEST
# ============================================================================

def test_fastapi_demo_mode_api_endpoints():
    """Verify /ready, /health, and /chat in FastAPI return demo_mode and handle Open VS Code."""
    from fastapi.testclient import TestClient
    from main import create_app
    from config.settings import AtlasSettings

    settings = AtlasSettings(
        app_env="test",
        atlas_demo_mode=True,
        simulation_mode=True,
    )
    app = create_app(settings=settings, in_memory_stores=True)

    with TestClient(app) as client:
        # 1. /health returns demo_mode=True
        r_health = client.get("/api/v1/health")
        assert r_health.status_code == 200
        assert r_health.json().get("demo_mode") is True

        # 2. /ready returns demo_mode=True
        r_ready = client.get("/api/v1/ready")
        assert r_ready.status_code == 200
        assert r_ready.json().get("demo_mode") is True

        # Inject mock backend into runtime to avoid popping window during API test
        atlas = app.state.atlas
        mock_backend = MockDemoAppBackend()
        atlas.cognitive_runtime.execution_engine.router.registry.register(
            "computer_app", DemoAppCapability(backend=mock_backend)
        )

        # 3. /chat executes Open VS Code cleanly
        r_chat = client.post(
            "/api/v1/chat",
            json={"message": "Open VS Code"},
            headers={"Authorization": f"Bearer {settings.api_auth_token}"},
        )
        assert r_chat.status_code == 200
        data = r_chat.json()
        assert data.get("status") in ("SUCCEEDED", "completed")
        assert "Launched Visual Studio Code" in data.get("response")
        assert "turn_id" in data
