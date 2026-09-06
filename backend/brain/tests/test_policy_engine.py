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
