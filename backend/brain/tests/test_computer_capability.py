import pytest
import time
from typing import Optional, Dict, Any, List

from core.models.computer import (
    ComputerAction,
    ComputerTarget,
    ComputerObservation,
    ScreenDimensions,
    TargetType,
    ALLOWED_SPECIAL_KEYS,
)
from core.models.tool_call import ToolCall
from core.models.result import Result
from core.models.task import Task
from core.models.plan import Plan
from core.models.verification import VerificationResult
from core.models.policy import (
    PolicyDecision,
    PolicyContext,
    PolicyResult,
    AutonomyLevel,
    RiskLevel,
)
from core.models.recovery import (
    ExecutionOutcome,
    RecoveryAction,
    FailureClassification,
    RecoveryLimits,
    RecoveryContext,
)
from computer.mock_backend import MockComputerBackend
from computer.computer_capability import ComputerCapability
from safety.policy_engine import (
    StandardPolicyEngine,
    ComputerObservationRule,
    ComputerLowRiskActionRule,
    ComputerSensitiveActionRule,
    DefaultDenyRule,
)
from tools.tool_orchestrator import ToolOrchestrator
from tools.capability_registry import CapabilityRegistry
from tools.executor import Executor
from brain.execution import StandardExecutionEngine
from brain.verification import StandardVerifier
from brain.recovery.recovery_engine import StandardRecoveryEngine


# ============================================================================
# A. DOMAIN MODELS
# ============================================================================

class TestComputerDomainModels:
    def test_valid_and_invalid_computer_actions(self):
        assert ComputerAction.SCREENSHOT == "screenshot"
        assert ComputerAction.CLICK == "click"
        assert ComputerAction.DOUBLE_CLICK == "double_click"
        assert ComputerAction.MOVE == "move"
        assert ComputerAction.TYPE == "type"
        assert ComputerAction.PRESS_KEY == "press_key"
        assert ComputerAction.SCROLL == "scroll"
        assert ComputerAction.WAIT == "wait"

        with pytest.raises(ValueError):
            ComputerAction("invalid_action")

    def test_valid_and_invalid_screen_dimensions(self):
        dims = ScreenDimensions(width=1920, height=1080)
        assert dims.contains(0, 0) is True
        assert dims.contains(1919, 1079) is True
        assert dims.contains(1920, 1080) is False  # 0-indexed exclusive upper bound
        assert dims.contains(-1, 500) is False

        with pytest.raises(ValueError):
            ScreenDimensions(width=0, height=1080)

        with pytest.raises(ValueError):
            ScreenDimensions(width=1920, height=-100)

    def test_valid_target_coordinate_and_region(self):
        screen = ScreenDimensions(width=1920, height=1080)

        valid_coord = ComputerTarget(target_type=TargetType.COORDINATE, x=500, y=300)
        ok, err = valid_coord.validate_bounds(screen)
        assert ok is True
        assert err is None

        valid_region = ComputerTarget(target_type=TargetType.REGION, region=(100, 100, 200, 200))
        ok, err = valid_region.validate_bounds(screen)
        assert ok is True
        assert err is None

    def test_invalid_target_coordinates_and_bounds_rejection(self):
        screen = ScreenDimensions(width=1920, height=1080)

        # Missing x or y
        t_missing = ComputerTarget(target_type=TargetType.COORDINATE, x=500, y=None)
        ok, err = t_missing.validate_bounds(screen)
        assert ok is False
        assert "requires both" in err

        # Out-of-bounds coordinates
        t_oob = ComputerTarget(target_type=TargetType.COORDINATE, x=2000, y=500)
        ok, err = t_oob.validate_bounds(screen)
        assert ok is False
        assert "outside screen bounds" in err

        # Negative coordinates
        t_neg = ComputerTarget(target_type=TargetType.COORDINATE, x=-10, y=500)
        ok, err = t_neg.validate_bounds(screen)
        assert ok is False
        assert "outside screen bounds" in err

    def test_observation_immutability(self):
        obs = ComputerObservation(
            timestamp=123456.0,
            screen_dimensions=ScreenDimensions(width=1920, height=1080),
            active_window_title="Browser",
            process_name="chrome.exe",
        )
        assert obs.screen_dimensions.width == 1920
        assert obs.active_window_title == "Browser"

        with pytest.raises(Exception):
            obs.timestamp = 999999.0  # type: ignore


# ============================================================================
# B. SCREENSHOT & OBSERVATION
# ============================================================================

class TestComputerScreenshotObservation:
    def test_successful_screenshot_observation(self):
        mock_backend = MockComputerBackend(
            active_window_title="Document Editor",
            process_name="editor.exe",
        )
        cap = ComputerCapability(backend=mock_backend)

        res = cap.execute(action="screenshot", call_id="c-1")
        assert res.success is True
        assert res.capability == "computer"
        assert res.action == "screenshot"
        assert res.call_id == "c-1"

        obs = res.data
        assert isinstance(obs, ComputerObservation)
        assert obs.screen_dimensions.width == 1920
        assert obs.active_window_title == "Document Editor"
        assert obs.process_name == "editor.exe"
        assert obs.screenshot_base64 is not None

    def test_screenshot_backend_failure_handling(self):
        mock_backend = MockComputerBackend()
        mock_backend.should_fail = True
        mock_backend.failure_message = "Virtual display disconnected."
        cap = ComputerCapability(backend=mock_backend)

        res = cap.execute(action="screenshot")
        assert res.success is False
        assert "Virtual display disconnected." in res.message


# ============================================================================
# C. MOUSE ACTIONS (MOVE, CLICK, DOUBLE_CLICK, SCROLL)
# ============================================================================

class TestComputerMouseActions:
    def test_valid_click_and_move_through_mock_backend(self):
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)

        # Move
        res_move = cap.execute(action="move", x=400, y=300)
        assert res_move.success is True
        assert mock_backend.cursor_x == 400
        assert mock_backend.cursor_y == 300

        # Click
        res_click = cap.execute(action="click", x=600, y=700, button="left")
        assert res_click.success is True
        assert mock_backend.cursor_x == 600
        assert mock_backend.cursor_y == 700
        assert mock_backend.history[-1] == {"action": "click", "x": 600, "y": 700, "button": "left"}

        # Double Click
        res_dbl = cap.execute(action="double_click", x=200, y=200)
        assert res_dbl.success is True
        assert mock_backend.history[-1] == {"action": "double_click", "x": 200, "y": 200, "button": "left"}

    def test_out_of_bounds_and_negative_click_rejected(self):
        mock_backend = MockComputerBackend(screen_dimensions=ScreenDimensions(width=1000, height=800))
        cap = ComputerCapability(backend=mock_backend)

        # Out of bounds x
        res_oob = cap.execute(action="click", x=1500, y=400)
        assert res_oob.success is False
        assert "outside screen bounds" in res_oob.message

        # Negative y
        res_neg = cap.execute(action="click", x=200, y=-5)
        assert res_neg.success is False
        assert "outside screen bounds" in res_neg.message

    def test_scroll_action(self):
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)

        res = cap.execute(action="scroll", amount=5, direction="down")
        assert res.success is True
        assert mock_backend.history[-1] == {"action": "scroll", "amount": 5, "direction": "down"}


# ============================================================================
# D. KEYBOARD ACTIONS & AUDIT SAFETY
# ============================================================================

class TestComputerKeyboardActions:
    def test_allowed_key_press(self):
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)

        res = cap.execute(action="press_key", key="enter")
        assert res.success is True
        assert mock_backend.history[-1] == {"action": "press_key", "key": "enter"}

        res_char = cap.execute(action="press_key", key="a")
        assert res_char.success is True
        assert mock_backend.history[-1] == {"action": "press_key", "key": "a"}

    def test_invalid_key_rejected(self):
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)

        res = cap.execute(action="press_key", key="fake_invalid_key_name")
        assert res.success is False
        assert "not an authorized" in res.message

    def test_typing_succeeds_and_never_logs_raw_text(self):
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)

        secret_text = "MySecretPassphrase123"
        res = cap.execute(action="type", text=secret_text)

        assert res.success is True
        # Verify message and output do NOT expose raw text
        assert secret_text not in res.message
        assert secret_text not in res.output
        assert "Typed 21 characters" in res.message
        assert res.data["text_length"] == 21
        # Mock backend captured the keystrokes internally
        assert mock_backend.history[-1]["text_length"] == 21


# ============================================================================
# E. POLICY & PERMISSION ENGINE
# ============================================================================

class TestComputerPolicyEngine:
    def test_safe_observation_allowed_by_policy(self):
        policy_engine = StandardPolicyEngine()

        tool_call_shot = ToolCall(capability="computer", action="screenshot")
        ctx = PolicyContext.from_tool_call(tool_call_shot)
        res_shot = policy_engine.evaluate(tool_call_shot, ctx)
        assert res_shot.is_allowed is True
        assert res_shot.rule_id == "RULE_COMPUTER_SCREENSHOT"

        tool_call_wait = ToolCall(capability="computer", action="wait", parameters={"seconds": 2})
        res_wait = policy_engine.evaluate(tool_call_wait, ctx)
        assert res_wait.is_allowed is True

    def test_low_risk_action_allowed_in_assisted_autonomy(self):
        policy_engine = StandardPolicyEngine()

        tool_call_click = ToolCall(capability="computer", action="click", parameters={"x": 100, "y": 200})
        ctx_assisted = PolicyContext(
            capability="computer",
            action="click",
            parameters={"x": 100, "y": 200},
            autonomy_level=AutonomyLevel.ASSISTED,
        )
        res = policy_engine.evaluate(tool_call_click, ctx_assisted)
        assert res.is_allowed is True
        assert res.rule_id == "RULE_COMPUTER_CLICK"

    def test_low_risk_action_asks_permission_in_manual_autonomy(self):
        policy_engine = StandardPolicyEngine()

        tool_call_click = ToolCall(capability="computer", action="click", parameters={"x": 100, "y": 200})
        ctx_manual = PolicyContext(
            capability="computer",
            action="click",
            parameters={"x": 100, "y": 200},
            autonomy_level=AutonomyLevel.MANUAL,
        )
        res = policy_engine.evaluate(tool_call_click, ctx_manual)
        assert res.requires_permission is True
        assert res.decision == PolicyDecision.ASK_PERMISSION
        assert res.rule_id == "RULE_MANUAL_AUTONOMY_COMPUTER_ACTION"

    def test_sensitive_keyboard_shortcut_requires_confirmation(self):
        policy_engine = StandardPolicyEngine()

        tool_call_altf4 = ToolCall(capability="computer", action="press_key", parameters={"key": "alt+f4"})
        ctx = PolicyContext.from_tool_call(tool_call_altf4)
        res = policy_engine.evaluate(tool_call_altf4, ctx)
        assert res.decision == PolicyDecision.REQUIRE_CONFIRMATION
        assert res.rule_id == "RULE_DANGEROUS_KEY_COMBINATION"

    def test_sensitive_text_typing_asks_permission(self):
        policy_engine = StandardPolicyEngine()

        tool_call_pwd = ToolCall(capability="computer", action="type", parameters={"text": "my_admin_password_123"})
        ctx = PolicyContext.from_tool_call(tool_call_pwd)
        res = policy_engine.evaluate(tool_call_pwd, ctx)
        assert res.decision == PolicyDecision.ASK_PERMISSION
        assert res.rule_id == "RULE_SENSITIVE_TEXT_TYPING"

    def test_unknown_computer_action_denied_by_default(self):
        policy_engine = StandardPolicyEngine()

        tool_call_unknown = ToolCall(capability="computer", action="format_drive")
        ctx = PolicyContext.from_tool_call(tool_call_unknown)
        res = policy_engine.evaluate(tool_call_unknown, ctx)
        assert res.is_denied is True
        assert res.rule_id == "RULE_DEFAULT_DENY"


# ============================================================================
# F. ORCHESTRATOR INTEGRATION
# ============================================================================

class TestComputerOrchestratorIntegration:
    def test_orchestrator_executes_computer_action_with_policy(self):
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)
        registry = CapabilityRegistry(capabilities={"computer": cap})
        policy_engine = StandardPolicyEngine()
        orchestrator = ToolOrchestrator(registry=registry, policy_engine=policy_engine)

        tool_call = ToolCall(capability="computer", action="screenshot")
        res = orchestrator.execute(tool_call)

        assert res.success is True
        assert res.capability == "computer"
        assert res.action == "screenshot"
        assert mock_backend.screenshot_count == 1

    def test_denied_computer_action_never_reaches_backend(self):
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)
        registry = CapabilityRegistry(capabilities={"computer": cap})

        # Policy engine that explicitly denies all computer clicks
        class DenyClickPolicy(StandardPolicyEngine):
            def evaluate(self, tool_call: ToolCall, context: Optional[PolicyContext] = None) -> PolicyResult:
                if tool_call.action == "click":
                    return PolicyResult.deny("click_blocked", "Clicking is blocked in test environment.")
                return PolicyResult.allow("allow_rule")

        orchestrator = ToolOrchestrator(registry=registry, policy_engine=DenyClickPolicy())

        tool_call = ToolCall(capability="computer", action="click", parameters={"x": 100, "y": 100})
        res = orchestrator.execute(tool_call)

        assert res.success is False
        assert "Execution denied by policy [click_blocked]" in res.message
        # Crucial check: backend click was NEVER called
        assert len(mock_backend.history) == 0


# ============================================================================
# G. ARCHITECTURAL SECURITY BOUNDARY
# ============================================================================

class TestComputerArchitecturalSecurityBoundary:
    def test_computer_capability_cannot_bypass_policy_via_generic_orchestration(self):
        """
        Verify that no caller can invoke computer capability through ToolOrchestrator
        without satisfying the PolicyEngine barrier.
        """
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)
        registry = CapabilityRegistry(capabilities={"computer": cap})

        class StrictBlockAllComputerPolicy(StandardPolicyEngine):
            def evaluate(self, tool_call: ToolCall, context: Optional[PolicyContext] = None) -> PolicyResult:
                if tool_call.capability == "computer":
                    return PolicyResult.deny("strict_computer_block", "All computer use is prohibited.")
                return PolicyResult.allow("allow_other")

        orchestrator = ToolOrchestrator(registry=registry, policy_engine=StrictBlockAllComputerPolicy())

        # Attempt multiple computer actions
        for act in ("screenshot", "click", "type", "press_key"):
            call = ToolCall(capability="computer", action=act, parameters={"x": 50, "y": 50, "text": "a", "key": "enter"})
            res = orchestrator.execute(call)
            assert res.success is False
            assert "strict_computer_block" in res.message

        assert len(mock_backend.history) == 0

    def test_legacy_tools_do_not_expose_computer_capability(self):
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)
        registry = CapabilityRegistry(capabilities={"computer": cap})
        policy_engine = StandardPolicyEngine()
        orchestrator = ToolOrchestrator(registry=registry, policy_engine=policy_engine)

        # Attempt to invoke a legacy desktop tool through orchestrator
        call_legacy = ToolCall(capability="calculator", action="open")
        res = orchestrator.execute(call_legacy)
        assert res.success is False
        assert "legacy_desktop_tool_restriction" in res.message or "unauthorized capability" in res.message.lower()


# ============================================================================
# H. VERIFICATION & TASK-SUCCESS DISTINCTION
# ============================================================================

class TestComputerVerification:
    def test_execution_success_distinct_from_task_success(self):
        """
        Low-level click execution success != overall task verification success.
        Demonstrates that clicking a target must still be verified through observation.
        """
        verifier = StandardVerifier()

        plan = Plan(
            goal="submit form",
            steps=[
                Task(id=1, type="computer", action="Computer Click", tool="computer", parameters={"action": "click", "x": 100, "y": 200}, status="completed"),
                Task(id=2, type="computer", action="Computer Screenshot", tool="computer", parameters={"action": "screenshot"}, status="failed"),
            ],
            status="failed",
        )

        results = [
            Result.ok(message="Clicked at (100, 200)", output="Clicked"),
            Result.fail(message="Verification observation failed to capture expected change"),
        ]

        ver_res = verifier.verify(plan, results)
        assert ver_res.verified is False
        assert ver_res.status == "failed"
        assert ver_res.failed_task_id == 2


# ============================================================================
# I. RECOVERY INTEGRATION
# ============================================================================

class TestComputerRecoveryIntegration:
    def test_transient_computer_failure_retries_and_succeeds(self):
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)
        registry = CapabilityRegistry(capabilities={"computer": cap})
        orchestrator = ToolOrchestrator(registry=registry)

        call_count = 0
        class RouterStub:
            def route(self, task: Task) -> Result:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return Result.fail(message="Display capture temporary timeout (503 transient)")
                return orchestrator.execute_task(task)

        exec_engine = StandardExecutionEngine(router=RouterStub())
        verifier = StandardVerifier()
        recovery_engine = StandardRecoveryEngine(limits=RecoveryLimits(max_attempts=3, max_retries_per_action=2))

        plan = Plan(
            goal="take desktop screenshot",
            steps=[
                Task(id=1, type="computer", action="Computer Screenshot", tool="computer", parameters={"action": "screenshot"}, status="pending")
            ],
            status="pending",
        )

        final_plan, results, ver, ctx = recovery_engine.recover(
            original_goal="take desktop screenshot",
            initial_plan=plan,
            execute_fn=exec_engine.execute,
            verify_fn=verifier.verify,
        )

        assert ver.verified is True
        assert ctx.attempt == 2
        assert ctx.outcome == ExecutionOutcome.SUCCESS
        assert call_count == 2
        assert mock_backend.screenshot_count == 1


# ============================================================================
# J. LOOP LIMITS
# ============================================================================

class TestComputerLoopBounds:
    def test_bounded_computer_action_cycle(self):
        """Ensure repeated failure terminates cleanly within configured limits."""
        def execute_fn(p: Plan) -> List[Result]:
            p.status = "failed"
            return [Result.fail(message="Target UI element unavailable")]

        def verify_fn(p: Plan, res: List[Result]) -> VerificationResult:
            return VerificationResult(verified=False, status="failed", confidence=1.0, reason="Failed.")

        limits = RecoveryLimits(max_attempts=3, max_replans=2)
        engine = StandardRecoveryEngine(limits=limits)

        plan = Plan(
            goal="click login button",
            steps=[
                Task(id=1, type="computer", action="Computer Click", tool="computer", parameters={"action": "click", "x": 50, "y": 50}, status="pending")
            ],
            status="pending",
        )

        final_plan, results, ver, ctx = engine.recover("click login button", plan, execute_fn, verify_fn)

        assert ver.verified is False
        assert ctx.attempt <= limits.max_attempts
        assert len(ctx.plan_history) <= limits.max_total_recovery_steps
