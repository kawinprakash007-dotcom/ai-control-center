"""Tests for Phase 3.3 Policy / Permission / Safety Engine."""

import pytest
from unittest.mock import MagicMock

from core.models.tool_call import ToolCall
from core.models.result import Result
from core.models.policy import (
    PolicyDecision,
    PolicyContext,
    PolicyResult,
    AutonomyLevel,
    RiskLevel,
)
from core.interfaces.policy_interface import PolicyEngineInterface
from safety.policy_engine import (
    StandardPolicyEngine,
    PolicyRule,
    ProhibitedCapabilityRule,
    LegacyToolRestrictionRule,
    DestructiveMemoryRule,
    SensitiveMemorySaveRule,
    SafeReadAndSearchRule,
    ChatResponseRule,
    DefaultDenyRule,
)
from tools.tool_orchestrator import ToolOrchestrator
from tools.capability_registry import CapabilityRegistry


class SpyCapability:
    """Capability spy for verifying execution count and parameters."""

    def __init__(self, output="Success output"):
        self.output = output
        self.call_count = 0
        self.last_task = None

    def __call__(self, task=None):
        self.call_count += 1
        self.last_task = task
        return self.output


# ============================================================================
# 1. ALLOW DECISIONS (SAFE / LOW-RISK)
# ============================================================================

class TestPolicyAllowDecisions:
    """Test safe read-only operations receive deterministic ALLOW decisions."""

    @pytest.fixture
    def engine(self):
        return StandardPolicyEngine()

    def test_web_search_allowed(self, engine):
        """Web search is permitted as a safe read-only operation."""
        tc = ToolCall(capability="web", action="search", parameters={"query": "python"})
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.ALLOW
        assert res.is_allowed is True
        assert "RULE_SAFE_WEB_SEARCH" in res.rule_id

    def test_web_fetch_allowed(self, engine):
        """Web fetch is permitted as a safe read-only operation."""
        tc = ToolCall(capability="web", action="fetch", parameters={"url": "https://example.com"})
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.ALLOW
        assert res.is_allowed is True

    def test_web_research_allowed(self, engine):
        """Bounded web research is permitted as a safe read-only operation."""
        tc = ToolCall(capability="web", action="research", parameters={"query": "benchmarks"})
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.ALLOW
        assert res.is_allowed is True

    def test_knowledge_query_allowed(self, engine):
        """Knowledge retrieval is permitted as a safe read-only operation."""
        tc = ToolCall(capability="knowledge", action="query", parameters={"query": "system architecture"})
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.ALLOW
        assert res.is_allowed is True

    def test_memory_read_allowed(self, engine):
        """Memory recall/read is permitted as a safe read-only operation."""
        tc = ToolCall(capability="memory", action="read", parameters={"key": "theme"})
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.ALLOW
        assert res.is_allowed is True

    def test_safe_memory_save_allowed(self, engine):
        """General non-sensitive preference save is permitted in assisted mode."""
        tc = ToolCall(capability="memory", action="save", parameters={"key": "theme", "value": "dark"})
        ctx = PolicyContext.from_tool_call(tc, autonomy_level=AutonomyLevel.ASSISTED)
        res = engine.evaluate(tc, ctx)
        assert res.decision == PolicyDecision.ALLOW
        assert res.is_allowed is True
        assert "RULE_SAFE_MEMORY_SAVE" in res.rule_id


# ============================================================================
# 2. DENY DECISIONS (FORBIDDEN, UNKNOWN, LEGACY)
# ============================================================================

class TestPolicyDenyDecisions:
    """Test forbidden, unknown, and restricted calls receive deterministic DENY decisions."""

    @pytest.fixture
    def engine(self):
        return StandardPolicyEngine()

    def test_prohibited_shell_capability_denied(self, engine):
        """Shell execution is strictly forbidden."""
        tc = ToolCall(capability="shell", action="run", parameters={"cmd": "whoami"})
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.DENY
        assert res.is_denied is True
        assert res.rule_id == "RULE_FORBIDDEN_CAPABILITY"

    def test_prohibited_subprocess_action_denied(self, engine):
        """Subprocess action is strictly forbidden."""
        tc = ToolCall(capability="web", action="subprocess", parameters={"cmd": "ls"})
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.DENY
        assert res.rule_id == "RULE_FORBIDDEN_CAPABILITY"

    def test_legacy_desktop_tool_denied(self, engine):
        """Legacy desktop tools are denied from generic model orchestration."""
        tc = ToolCall(capability="calculator", action="open", parameters={})
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.DENY
        assert res.rule_id == "RULE_LEGACY_TOOL_RESTRICTED"

    def test_unknown_capability_denied_by_default(self, engine):
        """Unknown or unwhitelisted capability triggers default deny."""
        tc = ToolCall(capability="quantum_simulator", action="run", parameters={})
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.DENY
        assert res.rule_id == "RULE_DEFAULT_DENY"

    def test_unknown_action_on_known_capability_denied(self, engine):
        """Unknown action on known capability triggers default deny."""
        tc = ToolCall(capability="web", action="drop_cache_table", parameters={})
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.DENY
        assert res.rule_id == "RULE_DEFAULT_DENY"

    def test_invalid_tool_call_denied(self, engine):
        """Non-ToolCall objects trigger safe DENY."""
        res = engine.evaluate("not a tool call")
        assert res.decision == PolicyDecision.DENY
        assert res.rule_id == "RULE_INVALID_TOOL_CALL"


# ============================================================================
# 3. ASK_PERMISSION & REQUIRE_CONFIRMATION DECISIONS
# ============================================================================

class TestPolicyPermissionAndConfirmationDecisions:
    """Test sensitive and destructive operations require permission or confirmation."""

    @pytest.fixture
    def engine(self):
        return StandardPolicyEngine()

    def test_sensitive_credential_key_save_asks_permission(self, engine):
        """Saving password or token keys triggers ASK_PERMISSION."""
        tc = ToolCall(
            capability="memory",
            action="save",
            parameters={"key": "github_api_key", "value": "ghp_secret123"},
        )
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.ASK_PERMISSION
        assert res.requires_permission is True
        assert res.rule_id == "RULE_MEMORY_SAVE_CREDENTIAL_KEY"
        assert "sensitive or credential-related" in res.reason

    def test_manual_autonomy_memory_save_asks_permission(self, engine):
        """Memory save under MANUAL autonomy level requires explicit permission."""
        tc = ToolCall(
            capability="memory",
            action="save",
            parameters={"key": "editor_theme", "value": "solarized"},
        )
        ctx = PolicyContext.from_tool_call(tc, autonomy_level=AutonomyLevel.MANUAL)
        res = engine.evaluate(tc, ctx)
        assert res.decision == PolicyDecision.ASK_PERMISSION
        assert res.rule_id == "RULE_MANUAL_AUTONOMY_MEMORY_SAVE"

    def test_destructive_memory_forget_requires_confirmation(self, engine):
        """Permanently forgetting a memory preference triggers REQUIRE_CONFIRMATION."""
        tc = ToolCall(
            capability="memory",
            action="forget",
            parameters={"key": "favorite_food"},
        )
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.REQUIRE_CONFIRMATION
        assert res.requires_permission is True
        assert res.rule_id == "RULE_DESTRUCTIVE_MEMORY_FORGET"
        assert "destructive" in res.reason.lower()


# ============================================================================
# 4. DEFAULT-DENY GUARANTEE & ERROR HANDLING
# ============================================================================

class TestDefaultDenyPrinciple:
    """Test that policy never fails open."""

    def test_rule_exception_triggers_default_deny(self):
        """If a policy rule raises an unhandled exception, engine safely DENIES."""
        class BuggyRule(PolicyRule):
            rule_id = "RULE_BUGGY"
            description = "Buggy rule for testing fail-safe behavior"

            def matches(self, tool_call, context):
                raise RuntimeError("Unexpected internal crash in rule matcher")

            def evaluate(self, tool_call, context):
                return PolicyResult.allow(self.rule_id)

        engine = StandardPolicyEngine(rules=[BuggyRule()])
        tc = ToolCall(capability="web", action="search", parameters={"query": "test"})
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.DENY
        assert res.rule_id == "RULE_POLICY_ENGINE_ERROR"
        assert "Default deny enforced" in res.reason


# ============================================================================
# 5. ORCHESTRATOR INTEGRATION & SECURITY BOUNDARY
# ============================================================================

class TestOrchestratorPolicyEnforcement:
    """Verify that ToolOrchestrator strictly enforces policy decisions."""

    def test_denied_call_never_reaches_capability(self):
        """A call denied by policy must never execute the underlying capability."""
        spy = SpyCapability(output="Should never be called")
        reg = CapabilityRegistry(capabilities={"web": spy})
        orchestrator = ToolOrchestrator(registry=reg)

        # Prohibited action
        tc = ToolCall(capability="web", action="shell", parameters={"cmd": "echo bad"})
        result = orchestrator.execute(tc)

        assert result.success is False
        assert spy.call_count == 0, "Security violation: capability was invoked despite policy rejection!"

    def test_permission_required_call_never_reaches_capability(self):
        """A call requiring permission does not execute without external approval."""
        spy = SpyCapability(output="Should never be called")
        reg = CapabilityRegistry(capabilities={"memory": spy})
        orchestrator = ToolOrchestrator(registry=reg)

        # Saving credential key requires permission
        tc = ToolCall(
            capability="memory",
            action="save",
            parameters={"key": "openai_api_key", "value": "sk-secret"},
        )
        result = orchestrator.execute(tc)

        assert result.success is False
        assert result.data is not None
        assert result.data.get("requires_permission") is True
        assert spy.call_count == 0, "Security violation: capability executed without permission!"

    def test_confirmation_required_call_never_reaches_capability(self):
        """A call requiring confirmation does not execute without explicit confirmation."""
        spy = SpyCapability(output="Should never be called")
        reg = CapabilityRegistry(capabilities={"memory": spy})
        orchestrator = ToolOrchestrator(registry=reg)

        # Destructive forget requires confirmation
        tc = ToolCall(
            capability="memory",
            action="forget",
            parameters={"key": "work_address"},
        )
        result = orchestrator.execute(tc)

        assert result.success is False
        assert result.data is not None
        assert result.data.get("requires_confirmation") is True
        assert spy.call_count == 0, "Security violation: capability executed without confirmation!"

    def test_allowed_call_executes_exactly_once(self):
        """An allowed call passes through and executes exactly once."""
        spy = SpyCapability(output="Executed cleanly")
        reg = CapabilityRegistry(capabilities={"web": spy})
        orchestrator = ToolOrchestrator(registry=reg)

        tc = ToolCall(
            capability="web",
            action="search",
            parameters={"query": "safe query"},
            call_id="call-valid-99",
        )
        result = orchestrator.execute(tc)

        assert result.success is True
        assert spy.call_count == 1
        assert result.output == "Executed cleanly"
        assert result.call_id == "call-valid-99"

    def test_architectural_security_boundary_custom_deny(self):
        """
        Architectural Boundary Proof:
        A registered, valid capability CANNOT be invoked through ToolOrchestrator
        if policy denies it, even when registered and whitelisted.
        """
        spy = SpyCapability(output="Secret capability output")
        reg = CapabilityRegistry(capabilities={"web": spy})

        # Custom policy engine that denies web search
        class StrictDenyWebEngine(PolicyEngineInterface):
            def evaluate(self, tool_call, context=None):
                return PolicyResult.deny(
                    rule_id="RULE_STRICT_WEB_LOCKDOWN",
                    reason="Web access is locked down by corporate security policy.",
                )

        orchestrator = ToolOrchestrator(
            registry=reg,
            policy_engine=StrictDenyWebEngine(),
        )

        tc = ToolCall(capability="web", action="search", parameters={"query": "test query"})
        result = orchestrator.execute(tc)

        assert result.success is False
        assert "Execution denied by policy [RULE_STRICT_WEB_LOCKDOWN]" in result.message
        assert spy.call_count == 0, "Security failure: lockdown policy was bypassed!"

    def test_policy_engine_failure_does_not_fail_open(self):
        """If policy evaluation crashes, orchestrator fails safe (denies execution)."""
        spy = SpyCapability()
        reg = CapabilityRegistry(capabilities={"web": spy})

        class CrashingPolicyEngine(PolicyEngineInterface):
            def evaluate(self, tool_call, context=None):
                raise RuntimeError("Critical memory corruption in policy evaluator")

        orchestrator = ToolOrchestrator(
            registry=reg,
            policy_engine=CrashingPolicyEngine(),
        )

        tc = ToolCall(capability="web", action="search", parameters={"query": "test query"})
        result = orchestrator.execute(tc)

        assert result.success is False
        assert "Policy evaluation error" in result.message
        assert spy.call_count == 0, "Fatal security flaw: crashed policy engine failed open!"


# ============================================================================
# 6. OBSERVABILITY & METADATA PROPAGATION
# ============================================================================

class TestPolicyObservabilityAndMetadata:
    """Test that policy metadata, reasons, and call_ids are preserved."""

    def test_metadata_preservation_on_denial(self):
        """Policy denial result carries structured rule_id and reason."""
        class CustomDenyEngine(PolicyEngineInterface):
            def evaluate(self, tool_call, context=None):
                return PolicyResult.deny(
                    rule_id="RULE_AUDIT_DENIAL",
                    reason="Denial for audit testing.",
                )

        orchestrator = ToolOrchestrator(policy_engine=CustomDenyEngine())
        tc = ToolCall(
            capability="web",
            action="search",
            parameters={"query": "observability test"},
            call_id="obs-call-123",
        )
        result = orchestrator.execute(tc)

        assert result.call_id == "obs-call-123"
        assert result.capability == "web"
        assert result.action == "search"
        assert result.data is not None
        assert "policy_result" in result.data
        assert result.data["policy_result"]["rule_id"] == "RULE_AUDIT_DENIAL"
        assert result.data["policy_result"]["reason"] == "Denial for audit testing."


# ============================================================================
# 7. CHAT CAPABILITY POLICY & STRICT SECURITY BOUNDARIES
# ============================================================================

class TestChatPolicyAndSecurityBoundary:
    """
    Focused tests for conversational chat response authorization and strict default-deny boundaries:
    1. chat/respond_user is allowed
    2. ordinary chat response does not require confirmation
    3. unknown chat action remains denied
    4. chat cannot execute device actions
    5. chat cannot invoke prohibited shell actions
    6. default-deny still works for unknown capabilities
    7. dangerous existing actions remain governed by existing policy
    8. no frontend/API bypass (ToolOrchestrator rejects unauthorized actions)
    """

    @pytest.fixture
    def engine(self):
        return StandardPolicyEngine()

    def test_01_chat_respond_user_is_allowed(self, engine):
        """1. chat/respond_user and canonical variants are allowed as safe conversational responses."""
        for action in ("respond_user", "respond user", "Respond to User", "respond to user", "respond-user", "respond_to_user"):
            tc = ToolCall(capability="chat", action=action, parameters={"query": "Hello"})
            res = engine.evaluate(tc)
            assert res.decision == PolicyDecision.ALLOW, f"Action '{action}' was not allowed"
            assert res.is_allowed is True
            assert res.rule_id == "RULE_CHAT_RESPONSE_ALLOWED"
            assert res.metadata.get("risk_level") == RiskLevel.SAFE.value

    def test_02_ordinary_chat_response_does_not_require_confirmation(self, engine):
        """2. Ordinary chat response does not require confirmation or user permission."""
        tc = ToolCall(capability="chat", action="respond_user", parameters={"query": "Check system status"})
        res = engine.evaluate(tc)
        assert res.decision == PolicyDecision.ALLOW
        assert res.requires_permission is False
        assert res.explanation is None

    def test_03_unknown_chat_action_remains_denied(self, engine):
        """3. Unknown or arbitrary actions on chat capability remain strictly denied by default-deny."""
        for action in ("arbitrary_action", "launch_missile", "reboot", "dance", "inject_code"):
            tc = ToolCall(capability="chat", action=action)
            res = engine.evaluate(tc)
            assert res.decision == PolicyDecision.DENY
            assert res.is_denied is True
            assert res.rule_id == "RULE_DEFAULT_DENY"
            assert "Default deny enforced" in res.reason

    def test_04_chat_cannot_execute_device_actions(self, engine):
        """4. Chat capability cannot execute device operations or dispatch to hardware."""
        for action in ("execute_device", "device_dispatch", "device_action", "fly_drone", "arm_rover"):
            tc = ToolCall(capability="chat", action=action)
            res = engine.evaluate(tc)
            assert res.decision == PolicyDecision.DENY
            assert res.is_denied is True
            assert res.rule_id in ("RULE_FORBIDDEN_CAPABILITY", "RULE_DEFAULT_DENY")

    def test_05_chat_cannot_invoke_prohibited_shell_actions(self, engine):
        """5. Chat capability cannot invoke shell, OS, powershell, or subprocess actions."""
        for action in ("shell", "exec", "eval", "subprocess", "cmd", "powershell", "system", "chmod"):
            tc = ToolCall(capability="chat", action=action)
            res = engine.evaluate(tc)
            assert res.decision == PolicyDecision.DENY
            assert res.is_denied is True
            assert res.rule_id == "RULE_FORBIDDEN_CAPABILITY"
            assert "strictly forbidden" in res.reason

    def test_06_default_deny_still_works_for_unknown_capabilities(self, engine):
        """6. Default-deny still catches completely unknown capabilities and actions."""
        for cap in ("teleportation", "quantum_telemetry", "drone_direct", "unregistered_bot"):
            tc = ToolCall(capability=cap, action="respond_user")
            res = engine.evaluate(tc)
            assert res.decision == PolicyDecision.DENY
            assert res.is_denied is True
            assert res.rule_id == "RULE_DEFAULT_DENY"

    def test_07_dangerous_existing_actions_remain_governed(self, engine):
        """7. Dangerous capabilities and actions remain strictly governed by existing safety policies."""
        # Shell / system execution is strictly forbidden
        tc_shell = ToolCall(capability="shell", action="execute", parameters={"cmd": "ls"})
        assert engine.evaluate(tc_shell).decision == PolicyDecision.DENY

        # Destructive unconfirmed memory forget requires confirmation
        tc_forget = ToolCall(capability="memory", action="forget", parameters={"key": "secret"})
        res_forget = engine.evaluate(tc_forget)
        assert res_forget.decision == PolicyDecision.REQUIRE_CONFIRMATION

        # Sensitive key save in manual autonomy mode requires permission
        tc_save = ToolCall(capability="memory", action="save", parameters={"key": "api_key", "value": "xyz"})
        ctx = PolicyContext.from_tool_call(tc_save, autonomy_level=AutonomyLevel.MANUAL)
        res_save = engine.evaluate(tc_save, ctx)
        assert res_save.decision == PolicyDecision.ASK_PERMISSION

    def test_08_no_frontend_or_api_bypass(self, engine):
        """8. ToolOrchestrator rejects unapproved actions even if registered and whitelisted."""
        spy = SpyCapability(output="Safe chat output")
        reg = CapabilityRegistry(capabilities={"chat": spy})
        orchestrator = ToolOrchestrator(
            registry=reg,
            policy_engine=engine,
            allowed_capabilities={"chat": {"respond_user", "shell", "arbitrary_exploit"}},
        )

        # Prohibited action via chat capability must fail (blocked by orchestrator safety guard or policy engine)
        tc_bad = ToolCall(capability="chat", action="shell", parameters={"cmd": "whoami"})
        res_bad = orchestrator.execute(tc_bad)
        assert res_bad.success is False
        assert "Prohibited capability or action" in res_bad.message or "Execution denied by policy" in res_bad.message
        assert spy.call_count == 0

        # Unknown action via chat capability must fail via PolicyEngine default-deny
        tc_unknown = ToolCall(capability="chat", action="arbitrary_exploit")
        res_unknown = orchestrator.execute(tc_unknown)
        assert res_unknown.success is False
        assert "Execution denied by policy [RULE_DEFAULT_DENY]" in res_unknown.message
        assert spy.call_count == 0

        # Safe chat action is permitted through ToolOrchestrator
        tc_safe = ToolCall(capability="chat", action="respond_user", parameters={"query": "Hello"})
        res_safe = orchestrator.execute(tc_safe)
        assert res_safe.success is True
        assert spy.call_count == 1

    def test_09_manual_command_check_drone_status_permitted(self, engine):
        """
        Exact manual verification command:
        'Check the drone status'
        Expected:
        - Request accepted
        - CognitiveRuntime processes it
        - Policy permits response generation (RULE_CHAT_RESPONSE_ALLOWED)
        - User receives an ATLAS response (no PolicyEngine denial)
        """
        from brain.request_understanding import StandardRequestUnderstanding
        from brain.decision_engine import StandardDecisionEngine
        from brain.planning import StandardPlanner
        from core.models.runtime import TurnStatus
        from runtime.cognitive_runtime import CognitiveRuntime
        from core.interfaces.execution_engine_interface import ExecutionEngineInterface

        # 1. Verify Request Understanding and Decision
        req = StandardRequestUnderstanding().understand("Check the drone status")
        assert req.original_text == "Check the drone status"
        dec = StandardDecisionEngine().decide(req)
        plan = StandardPlanner().plan(dec)

        # 2. Verify Plan contains live_state / chat query
        assert len(plan.steps) >= 1
        step = plan.steps[0]
        assert step.tool in ("live_state", "chat") or step.type in ("device", "chat")

        # 3. Verify Policy permits the proposed tool call
        tc = ToolCall(capability=step.tool, action=step.action)
        ctx = PolicyContext(capability=tc.capability, action=tc.action)
        res = engine.evaluate(tc, ctx)
        assert res.decision == PolicyDecision.ALLOW
        assert res.is_allowed is True

        # 4. Verify CognitiveRuntime turn execution
        class StubExecutionEngine(ExecutionEngineInterface):
            def execute(self, p):
                for task in p.steps:
                    task.status = "completed"
                    task.result = "ATLAS: Drone telemetry is nominal."
                p.status = "completed"
                return [Result(success=True, message="Responded", output="ATLAS: Drone telemetry is nominal.")]

        runtime = CognitiveRuntime(
            policy_engine=engine,
            execution_engine=StubExecutionEngine(),
        )
        turn_result = runtime.execute_turn("Check the drone status")
        assert turn_result.status == TurnStatus.SUCCEEDED
        assert "Policy denied execution" not in turn_result.response
        assert "Drone telemetry is nominal" in turn_result.response
