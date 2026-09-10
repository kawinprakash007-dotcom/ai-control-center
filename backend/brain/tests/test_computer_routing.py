import os
import sys
import subprocess
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest

from core.models.request import Request
from core.models.tool_call import ToolCall
from core.models.policy import PolicyContext, RiskLevel
from core.models.computer import ComputerAction
from core.models.runtime import TurnStatus
from brain.decision_engine import StandardDecisionEngine
from brain.planning import StandardPlanner
from safety.policy_engine import StandardPolicyEngine
from computer.computer_capability import ComputerCapability
from computer.mock_backend import MockComputerBackend
from runtime.cognitive_runtime import CognitiveRuntime


def make_request(text: str) -> Request:
    return Request(
        id="test-req-001",
        original_text=text,
        normalized_text=text,
        session_id="test-session",
        timestamp=datetime.now(),
        parameters={},
        constraints={},
    )


# ============================================================================
# 1. "open VS Code" does NOT produce teh.execute.tool
# ============================================================================

def test_open_vscode_does_not_produce_teh_execute_tool():
    engine = StandardDecisionEngine()
    planner = StandardPlanner()

    for cmd in [
        "open the vs code in my pc",
        "open teh vs code in my pc",
        "open VS Code",
        "launch VS Code",
        "start Visual Studio Code",
        "open vscode",
    ]:
        req = make_request(cmd)
        decision = engine.decide(req)

        # 1. Decision must NOT treat 'teh', 'the', or 'vs' as a tool
        assert decision.routing_hints.get("tool_hint") != "teh", f"Failed for {cmd}"
        assert decision.routing_hints.get("tool_hint") != "the", f"Failed for {cmd}"
        assert decision.routing_hints.get("tool_hint") != "vs", f"Failed for {cmd}"

        # 2. Decision must route to controlled computer action
        assert decision.primary_goal == "computer_action", f"Failed for {cmd}"
        assert decision.routing_hints.get("tool_hint") == "computer", f"Failed for {cmd}"

        # 3. Plan must NOT create a task for tool 'teh', 'the', or 'vs' with 'Execute Tool'
        plan = planner.plan(decision)
        assert len(plan.steps) >= 1
        for step in plan.steps:
            tool_name = getattr(step, "tool", None)
            action_name = getattr(step, "action", None)
            assert tool_name not in ("teh", "the", "vs"), f"Step tool invalid for {cmd}: {tool_name}"
            assert action_name != "Execute Tool", f"Step action must not be Execute Tool for {cmd}"
            assert tool_name == "computer", f"Step tool must be computer for {cmd}"


# ============================================================================
# 2. "open VS Code" does NOT produce shell / subprocess execution
# ============================================================================

def test_open_vscode_does_not_produce_shell_or_subprocess(monkeypatch):
    # Strictly assert that no shell or subprocess call is made
    mock_run = MagicMock(side_effect=RuntimeError("Subprocess run invoked!"))
    mock_popen = MagicMock(side_effect=RuntimeError("Subprocess popen invoked!"))
    mock_system = MagicMock(side_effect=RuntimeError("os.system invoked!"))

    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(subprocess, "Popen", mock_popen)
    monkeypatch.setattr(os, "system", mock_system)

    engine = StandardDecisionEngine()
    planner = StandardPlanner()
    policy = StandardPolicyEngine()
    runtime = CognitiveRuntime(
        decision_engine=engine,
        planner=planner,
        policy_engine=policy,
        memory_service=MagicMock(),
    )

    for cmd in [
        "open the vs code in my pc",
        "open teh vs code in my pc",
        "open VS Code",
        "launch VS Code",
        "start Visual Studio Code",
    ]:
        turn_result = runtime.execute_turn(cmd)
        assert mock_run.call_count == 0
        assert mock_popen.call_count == 0
        assert mock_system.call_count == 0
        # Must not produce teh.execute.tool error
        assert "teh.execute.tool" not in turn_result.response
        assert "teh.Execute Tool" not in turn_result.response


# ============================================================================
# 3. Invalid execute capabilities remain denied
# ============================================================================

def test_invalid_execute_capabilities_remain_denied():
    policy = StandardPolicyEngine()

    invalid_calls = [
        ToolCall(capability="teh", action="execute.tool"),
        ToolCall(capability="teh", action="Execute Tool"),
        ToolCall(capability="the", action="Execute Tool"),
        ToolCall(capability="vs", action="Execute Tool"),
        ToolCall(capability="shell", action="run"),
        ToolCall(capability="cmd", action="exec"),
        ToolCall(capability="powershell", action="execute"),
    ]

    for tc in invalid_calls:
        ctx = PolicyContext.from_tool_call(tc)
        res = policy.evaluate(tc, ctx)
        assert res.is_denied is True, f"Expected denial for {tc.capability}.{tc.action}"
        assert res.metadata.get("risk_level") == RiskLevel.CRITICAL.value


# ============================================================================
# 4. Normal chat response remains allowed
# ============================================================================

def test_normal_chat_response_remains_allowed():
    policy = StandardPolicyEngine()

    chat_calls = [
        ToolCall(capability="chat", action="respond_user"),
        ToolCall(capability="chat", action="respond user"),
        ToolCall(capability="chat", action="respond to user"),
        ToolCall(capability="chat", action="respond_to_user"),
    ]

    for tc in chat_calls:
        ctx = PolicyContext.from_tool_call(tc)
        res = policy.evaluate(tc, ctx)
        assert res.is_allowed is True
        assert res.rule_id == "RULE_CHAT_RESPONSE_ALLOWED"


# ============================================================================
# 5. Controlled computer-use requests use ComputerAction
# ============================================================================

def test_controlled_computer_use_requests_use_computer_action():
    engine = StandardDecisionEngine()

    controlled_inputs = [
        ("computer screenshot", "screenshot"),
        ("computer click", "click"),
        ("double click icon", "double_click"),
        ("move mouse to window", "move"),
        ("scroll down", "scroll"),
        ("wait for 5 seconds", "wait"),
    ]

    for user_input, expected_action in controlled_inputs:
        req = make_request(user_input)
        decision = engine.decide(req)

        assert decision.primary_goal == "computer_action", f"Failed for {user_input}"
        assert decision.routing_hints.get("tool_hint") == "computer", f"Failed for {user_input}"
        assert decision.routing_hints.get("action") == expected_action, f"Failed for {user_input}"
        # Validate that the action is a valid ComputerAction enum value
        assert ComputerAction(expected_action) is not None


# ============================================================================
# 6. Policy is still consulted
# ============================================================================

def test_policy_is_still_consulted():
    engine = StandardDecisionEngine()
    planner = StandardPlanner()

    mock_policy = MagicMock(spec=StandardPolicyEngine)
    from core.models.policy import PolicyResult
    mock_policy.evaluate.return_value = PolicyResult.deny(
        rule_id="RULE_TEST_POLICY_CONSULTED",
        reason="Test policy successfully intercepted and consulted.",
    )

    runtime = CognitiveRuntime(
        decision_engine=engine,
        planner=planner,
        policy_engine=mock_policy,
        memory_service=MagicMock(),
    )

    turn_res = runtime.execute_turn("open VS Code")

    # Verify policy evaluate was called with a ToolCall targeting 'computer'
    assert mock_policy.evaluate.called
    call_args = mock_policy.evaluate.call_args[0]
    tool_call = call_args[0]
    assert tool_call.capability == "computer"
    assert tool_call.action == "open_app"
    assert turn_res.status == TurnStatus.FAILED
    assert "RULE_TEST_POLICY_CONSULTED" in turn_res.response or "intercepted" in turn_res.response


# ============================================================================
# 7. Unsupported computer actions fail safely
# ============================================================================

def test_unsupported_computer_actions_fail_safely():
    mock_backend = MockComputerBackend()
    cap = ComputerCapability(backend=mock_backend)

    unsupported_actions = ["open_app", "launch_app", "run_binary", "exec_cmd", "arbitrary_script"]

    for act in unsupported_actions:
        res = cap.execute(action=act)
        assert res.success is False
        assert "Unsupported computer action" in res.message
        assert act in res.message
        # Crucial check: mock backend operations were never called
        assert len(mock_backend.history) == 0
        assert mock_backend.screenshot_count == 0


# ============================================================================
# 8. Unknown action remains default-denied
# ============================================================================

def test_unknown_action_remains_default_denied():
    policy = StandardPolicyEngine()

    unknown_actions = [
        ToolCall(capability="computer", action="format_drive"),
        ToolCall(capability="computer", action="reboot"),
        ToolCall(capability="computer", action="open_app"),
        ToolCall(capability="computer", action="launch_app"),
    ]

    for tc in unknown_actions:
        ctx = PolicyContext.from_tool_call(tc)
        res = policy.evaluate(tc, ctx)
        assert res.is_denied is True
        assert res.rule_id == "RULE_DEFAULT_DENY"


# ============================================================================
# 9. No arbitrary application execution path is introduced
# ============================================================================

def test_no_arbitrary_application_execution_path():
    engine = StandardDecisionEngine()

    malicious_inputs = [
        "open malicious.bat in my pc",
        "launch exploit.exe on desktop",
        "start payload.sh in my pc",
        "run trojan on computer",
    ]

    for cmd in malicious_inputs:
        req = make_request(cmd)
        decision = engine.decide(req)

        # Must route to computer.open_app, never creating a tool from the filename
        assert decision.routing_hints.get("tool_hint") == "computer"
        assert decision.routing_hints.get("action") == "open_app"
        assert decision.primary_goal == "computer_action"
        assert decision.routing_hints.get("tool_hint") not in ("malicious.bat", "exploit.exe", "payload.sh")


# ============================================================================
# 10. No frontend direct OS control exists
# ============================================================================

def test_no_frontend_direct_os_control():
    # Production security invariant: backend API routes expose no raw OS execution endpoints
    import inspect
    import core.api_routes as routes

    source = inspect.getsource(routes)
    forbidden_terms = ["subprocess", "os.system", "shell=True", "CreateProcess", "powershell", "cmd.exe"]

    for term in forbidden_terms:
        assert term not in source, f"Forbidden term '{term}' found in api_routes source code."


# ============================================================================
# 11. Deterministic variations for VS Code
# ============================================================================

def test_deterministic_variations_for_vscode():
    engine = StandardDecisionEngine()
    planner = StandardPlanner()

    variations = [
        "open VS Code",
        "launch VS Code",
        "start Visual Studio Code",
        "open the vs code in my pc",
        "open teh vs code in my pc",
        "open vscode",
        "start vscode in my pc",
    ]

    for cmd in variations:
        req = make_request(cmd)
        decision = engine.decide(req)

        assert decision.primary_goal == "computer_action", f"Failed for variation '{cmd}'"
        assert decision.routing_hints.get("tool_hint") == "computer", f"Failed for variation '{cmd}'"
        assert decision.routing_hints.get("action") == "open_app", f"Failed for variation '{cmd}'"
        assert decision.routing_hints.get("target") == "vscode", f"Failed for variation '{cmd}'"

        plan = planner.plan(decision)
        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert step.tool == "computer"
        assert step.parameters.get("action") == "open_app"
        assert step.parameters.get("target") == "vscode"
