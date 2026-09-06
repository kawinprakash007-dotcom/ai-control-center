import pytest
from typing import List, Optional, Any, Dict
from datetime import datetime

from core.models.request import Request
from core.models.decision import Decision, CapabilityType, ExecutionMode
from core.models.plan import Plan
from core.models.task import Task
from core.models.result import Result
from core.models.verification import VerificationResult
from core.models.pipeline import PipelineResult
from core.models.tool_call import ToolCall
from core.models.policy import (
    PolicyDecision,
    PolicyContext,
    PolicyResult,
)
from core.models.recovery import (
    ExecutionOutcome,
    RecoveryAction,
    FailureClassification,
    RecoveryLimits,
    RecoveryContext,
    RecoveryDecision,
    PlanHistoryEntry,
)
from core.interfaces.policy_interface import PolicyEngineInterface
from brain.recovery.failure_classifier import FailureClassifier
from brain.recovery.recovery_planner import (
    StandardRecoveryPlanner,
    compute_plan_signature,
)
from brain.recovery.recovery_engine import StandardRecoveryEngine
from brain.pipeline import StandardPipeline
from brain.execution import StandardExecutionEngine
from brain.verification import StandardVerifier
from tools.tool_orchestrator import ToolOrchestrator
from tools.capability_registry import CapabilityRegistry
from tools.executor import Executor


# ============================================================================
# TEST STUBS & SPY HELPERS
# ============================================================================

class FlakyMockRouter:
    """Mock router that simulates transient failures, then success on retry."""

    def __init__(self, failure_count: int = 1, failure_message: str = "Connection reset (503 timeout)"):
        self.failure_count = failure_count
        self.failure_message = failure_message
        self.calls = 0

    def route(self, task: Task) -> Result:
        self.calls += 1
        if self.calls <= self.failure_count:
            return Result.fail(
                message=self.failure_message,
                capability=getattr(task, "tool", "mock"),
                action=getattr(task, "action", "mock_action"),
            )
        return Result.ok(
            message="Operation completed successfully.",
            output="Recovered Output Data",
            capability=getattr(task, "tool", "mock"),
            action=getattr(task, "action", "mock_action"),
        )


class DeterministicSequenceRouter:
    """Mock router that returns a sequence of results based on task query or attempt."""

    def __init__(self, responses_by_query: Dict[str, Result]):
        self.responses_by_query = responses_by_query
        self.routed_tasks: List[Task] = []

    def route(self, task: Task) -> Result:
        self.routed_tasks.append(task)
        q = (task.parameters or {}).get("query", "")
        if q in self.responses_by_query:
            return self.responses_by_query[q]
        tool_act = f"{task.tool}.{task.action}"
        if tool_act in self.responses_by_query:
            return self.responses_by_query[tool_act]
        return Result.fail(message=f"No mock response for query '{q}' or '{tool_act}'")


# ============================================================================
# A. SUCCESS
# ============================================================================

class TestRecoverySuccess:
    def test_successful_execution_does_not_replan(self):
        plan = Plan(
            goal="explain AI",
            steps=[
                Task(id=1, type="web", action="Web Search", tool="web", parameters={"query": "AI"}, status="pending")
            ],
            status="pending",
        )
        engine = StandardRecoveryEngine()

        def execute_fn(p: Plan) -> List[Result]:
            p.status = "completed"
            p.steps[0].status = "completed"
            return [Result.ok(output="AI explanation")]

        def verify_fn(p: Plan, res: List[Result]) -> VerificationResult:
            return VerificationResult(verified=True, status="verified", confidence=1.0, reason="All planned tasks completed.")

        final_plan, results, ver, ctx = engine.recover("explain AI", plan, execute_fn, verify_fn)

        assert ver.verified is True
        assert ctx.outcome == ExecutionOutcome.SUCCESS
        assert ctx.attempt == 1
        assert ctx.replan_count == 0
        assert len(ctx.plan_history) == 0  # No failure history needed on immediate success


# ============================================================================
# B. RETRY (TRANSIENT FAILURE)
# ============================================================================

class TestRecoveryRetry:
    def test_transient_failure_retries_and_succeeds_within_bounds(self):
        flaky_router = FlakyMockRouter(failure_count=1, failure_message="Connection reset by peer (503 transient)")
        exec_engine = StandardExecutionEngine(router=flaky_router)
        verifier = StandardVerifier()

        plan = Plan(
            goal="fetch weather",
            steps=[
                Task(id=1, type="web", action="Web Search", tool="web", parameters={"query": "weather"}, status="pending")
            ],
            status="pending",
        )

        recovery_engine = StandardRecoveryEngine(limits=RecoveryLimits(max_attempts=3, max_retries_per_action=2))

        final_plan, results, ver, ctx = recovery_engine.recover(
            original_goal="fetch weather",
            initial_plan=plan,
            execute_fn=exec_engine.execute,
            verify_fn=verifier.verify,
        )

        assert ver.verified is True
        assert flaky_router.calls == 2
        assert ctx.attempt == 2
        assert ctx.replan_count == 0
        assert ctx.outcome == ExecutionOutcome.SUCCESS


# ============================================================================
# C. REPLAN
# ============================================================================

class TestRecoveryReplan:
    def test_failed_strategy_produces_new_strategy_targeting_original_goal(self):
        responses = {
            "astronomy": Result.fail(message="No evidence found: insufficient search results"),
            "astronomy documentation": Result.ok(output="Comprehensive astronomy guide"),
        }
        router = DeterministicSequenceRouter(responses)
        exec_engine = StandardExecutionEngine(router=router)
        verifier = StandardVerifier()

        initial_plan = Plan(
            goal="astronomy",
            steps=[
                Task(id=1, type="web", action="Web Search", tool="web", parameters={"query": "astronomy"}, status="pending")
            ],
            status="pending",
        )

        recovery_engine = StandardRecoveryEngine()

        final_plan, results, ver, ctx = recovery_engine.recover(
            original_goal="astronomy",
            initial_plan=initial_plan,
            execute_fn=exec_engine.execute,
            verify_fn=verifier.verify,
        )

        assert ver.verified is True
        assert ctx.replan_count == 1
        assert ctx.original_goal == "astronomy"
        assert len(router.routed_tasks) == 2
        assert router.routed_tasks[0].parameters["query"] == "astronomy"
        assert router.routed_tasks[1].parameters["query"] == "astronomy documentation"


# ============================================================================
# D. PARTIAL SUCCESS
# ============================================================================

class TestRecoveryPartialSuccess:
    def test_partial_success_preserves_successful_results(self):
        # Step 1 succeeds, Step 2 fails on first attempt, then succeeds on replan
        attempt = 0

        def execute_fn(p: Plan) -> List[Result]:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                p.steps[0].status = "completed"
                p.steps[1].status = "failed"
                p.status = "failed"
                return [
                    Result.ok(output="Part 1 retrieved successfully"),
                    Result.fail(message="Part 2 fetch error"),
                ]
            else:
                p.steps[0].status = "completed"
                p.status = "completed"
                return [Result.ok(output="Part 2 recovered via alternative")]

        def verify_fn(p: Plan, res: List[Result]) -> VerificationResult:
            if any(not getattr(r, "success", True) for r in res):
                return VerificationResult(verified=False, status="failed", confidence=0.5, reason="Step 2 failed")
            return VerificationResult(verified=True, status="verified", confidence=1.0, reason="All tasks complete")

        plan = Plan(
            goal="multi-step fetch",
            steps=[
                Task(id=1, type="web", action="Web Search", tool="web", parameters={"query": "p1"}, status="pending"),
                Task(id=2, type="web", action="Web Fetch", tool="web", parameters={"url": "http://p2.com"}, status="pending"),
            ],
            status="pending",
        )

        engine = StandardRecoveryEngine()
        final_plan, results, ver, ctx = engine.recover("multi-step fetch", plan, execute_fn, verify_fn)

        assert ver.verified is True
        assert len(results) >= 2
        assert any("Part 1 retrieved" in (r.output or "") for r in results)
        assert any("Part 2 recovered" in (r.output or "") for r in results)


# ============================================================================
# E. POLICY BLOCK
# ============================================================================

class TestRecoveryPolicyBlock:
    def test_policy_denied_never_retries_and_falls_back_or_aborts(self):
        plan = Plan(
            goal="system inspection",
            steps=[
                Task(id=1, type="tool", action="Execute Tool", tool="desktop_open_application", parameters={"app_name": "cmd"}, status="pending")
            ],
            status="pending",
        )

        def execute_fn(p: Plan) -> List[Result]:
            p.status = "failed"
            p.steps[0].status = "failed"
            return [
                Result.fail(
                    message="Execution denied by policy [legacy_desktop_tool_restriction]: Legacy desktop tools cannot be invoked.",
                    data={"policy_result": {"decision": "deny", "rule_id": "legacy_desktop_tool_restriction"}},
                )
            ]

        def verify_fn(p: Plan, res: List[Result]) -> VerificationResult:
            return VerificationResult(verified=False, status="failed", confidence=1.0, reason="Execution denied by policy.")

        engine = StandardRecoveryEngine(limits=RecoveryLimits(max_attempts=3, max_replans=2))
        final_plan, results, ver, ctx = engine.recover("system inspection", plan, execute_fn, verify_fn)

        assert ver.verified is False
        assert ctx.outcome == ExecutionOutcome.POLICY_BLOCKED
        # Should NOT retry the denied desktop action
        assert ctx.action_retry_counts.get("desktop_open_application.Execute Tool", 0) == 0


# ============================================================================
# F. REPETITION PROTECTION (ANTI-LOOP)
# ============================================================================

class TestRecoveryAntiLoop:
    def test_identical_failed_strategy_is_rejected_by_anti_loop(self):
        class CyclicPlanner(StandardRecoveryPlanner):
            def create_replan(self, context: RecoveryContext) -> Optional[Plan]:
                # Attempt to return the exact same plan step
                return Plan(
                    goal=context.original_goal,
                    steps=[
                        Task(id=1, type="web", action="Web Search", tool="web", parameters={"query": "loop"}, status="pending")
                    ],
                    status="pending",
                )

        plan = Plan(
            goal="loop test",
            steps=[
                Task(id=1, type="web", action="Web Search", tool="web", parameters={"query": "loop"}, status="pending")
            ],
            status="pending",
        )

        def execute_fn(p: Plan) -> List[Result]:
            p.status = "failed"
            return [Result.fail(message="Permanent failure")]

        def verify_fn(p: Plan, res: List[Result]) -> VerificationResult:
            return VerificationResult(verified=False, status="failed", confidence=1.0, reason="Execution failed.")

        engine = StandardRecoveryEngine(planner=CyclicPlanner(), limits=RecoveryLimits(max_attempts=10, max_replans=10))
        final_plan, results, ver, ctx = engine.recover("loop test", plan, execute_fn, verify_fn)

        # Anti-loop prevents loop from cycling up to 10 attempts
        assert ctx.attempt < 10
        assert ver.verified is False


# ============================================================================
# G. BUDGET ENFORCEMENT
# ============================================================================

class TestRecoveryBudget:
    def test_max_attempts_enforced(self):
        plan = Plan(
            goal="budget test",
            steps=[Task(id=1, type="web", action="Web Search", tool="web", parameters={"query": "q"}, status="pending")],
            status="pending",
        )

        attempts_executed = 0

        def execute_fn(p: Plan) -> List[Result]:
            nonlocal attempts_executed
            attempts_executed += 1
            p.status = "failed"
            return [Result.fail(message="Temporary failure timeout")]

        def verify_fn(p: Plan, res: List[Result]) -> VerificationResult:
            return VerificationResult(verified=False, status="failed", confidence=1.0, reason="Failed.")

        limits = RecoveryLimits(max_attempts=2, max_retries_per_action=5)
        engine = StandardRecoveryEngine(limits=limits)

        final_plan, results, ver, ctx = engine.recover("budget test", plan, execute_fn, verify_fn)

        assert attempts_executed <= 2
        assert ctx.attempt == 2
        assert ver.verified is False

    def test_max_replans_enforced(self):
        plan = Plan(
            goal="replan budget",
            steps=[Task(id=1, type="web", action="Web Search", tool="web", parameters={"query": "q1"}, status="pending")],
            status="pending",
        )

        replan_calls = 0

        def execute_fn(p: Plan) -> List[Result]:
            nonlocal replan_calls
            replan_calls += 1
            p.status = "failed"
            return [Result.fail(message="General error")]

        def verify_fn(p: Plan, res: List[Result]) -> VerificationResult:
            return VerificationResult(verified=False, status="failed", confidence=1.0, reason="Failed.")

        limits = RecoveryLimits(max_attempts=10, max_replans=1)
        engine = StandardRecoveryEngine(limits=limits)

        final_plan, results, ver, ctx = engine.recover("replan budget", plan, execute_fn, verify_fn)

        assert ctx.replan_count <= 1
        assert ver.verified is False


# ============================================================================
# H. ASK USER (ESCALATION)
# ============================================================================

class TestRecoveryAskUser:
    def test_sensitive_action_requires_user_permission_without_autonomous_bypass(self):
        plan = Plan(
            goal="save sensitive key",
            steps=[
                Task(id=1, type="memory", action="Manage Memory", tool="memory", parameters={"action": "save", "key": "user_api_key"}, status="pending")
            ],
            status="pending",
        )

        def execute_fn(p: Plan) -> List[Result]:
            p.status = "failed"
            return [
                Result.fail(
                    message="Permission required for 'memory.save' [sensitive_memory_save]: Key contains sensitive keyword 'api_key'.",
                    data={"requires_permission": True, "policy_result": {"decision": "ask_permission"}},
                )
            ]

        def verify_fn(p: Plan, res: List[Result]) -> VerificationResult:
            return VerificationResult(verified=False, status="failed", confidence=1.0, reason="Permission required for 'memory.save'")

        engine = StandardRecoveryEngine()
        final_plan, results, ver, ctx = engine.recover("save sensitive key", plan, execute_fn, verify_fn)

        assert ver.verified is False
        assert ctx.failure_classification == FailureClassification.PERMISSION_REQUIRED
        assert len(ctx.plan_history) == 1
        assert ctx.plan_history[0].recovery_action == RecoveryAction.ASK_USER
        assert ctx.metadata.get("requires_user_interaction") is True


# ============================================================================
# I. ABORT (UNRECOVERABLE FAILURE)
# ============================================================================

class TestRecoveryAbort:
    def test_unrecoverable_failure_terminates_cleanly(self):
        plan = Plan(
            goal="unrecoverable",
            steps=[Task(id=1, type="tool", action="Execute Tool", tool="unknown_tool", parameters={}, status="pending")],
            status="pending",
        )

        def execute_fn(p: Plan) -> List[Result]:
            p.status = "failed"
            return [Result.fail(message="No capability found for 'unknown_tool'")]

        def verify_fn(p: Plan, res: List[Result]) -> VerificationResult:
            return VerificationResult(verified=False, status="failed", confidence=1.0, reason="No capability found.")

        engine = StandardRecoveryEngine()
        final_plan, results, ver, ctx = engine.recover("unrecoverable", plan, execute_fn, verify_fn)

        assert ver.verified is False
        assert len(ctx.plan_history) >= 1
        assert ctx.plan_history[-1].recovery_action == RecoveryAction.ABORT


# ============================================================================
# J. ORIGINAL GOAL INVARIANT
# ============================================================================

class TestRecoveryOriginalGoal:
    def test_original_goal_remains_unchanged_across_replanning(self):
        responses = {
            "initial": Result.fail(message="Empty search"),
            "initial documentation": Result.ok(output="Final content"),
        }
        router = DeterministicSequenceRouter(responses)
        exec_engine = StandardExecutionEngine(router=router)
        verifier = StandardVerifier()

        initial_plan = Plan(
            goal="initial",
            steps=[Task(id=1, type="web", action="Web Search", tool="web", parameters={"query": "initial"}, status="pending")],
            status="pending",
        )

        engine = StandardRecoveryEngine()
        final_plan, results, ver, ctx = engine.recover(
            original_goal="What is quantum computing?",
            initial_plan=initial_plan,
            execute_fn=exec_engine.execute,
            verify_fn=verifier.verify,
        )

        assert ctx.original_goal == "What is quantum computing?"
        assert ver.verified is True


# ============================================================================
# K. PLAN HISTORY
# ============================================================================

class TestRecoveryPlanHistory:
    def test_plan_history_recorded_and_bounded(self):
        responses = {
            "q": Result.fail(message="insufficient evidence"),
            "q documentation": Result.fail(message="insufficient evidence"),
        }
        router = DeterministicSequenceRouter(responses)
        exec_engine = StandardExecutionEngine(router=router)
        verifier = StandardVerifier()

        initial_plan = Plan(
            goal="q",
            steps=[Task(id=1, type="web", action="Web Search", tool="web", parameters={"query": "q"}, status="pending")],
            status="pending",
        )

        engine = StandardRecoveryEngine(limits=RecoveryLimits(max_attempts=3, max_replans=2))
        final_plan, results, ver, ctx = engine.recover(
            original_goal="q",
            initial_plan=initial_plan,
            execute_fn=exec_engine.execute,
            verify_fn=verifier.verify,
        )

        assert len(ctx.plan_history) >= 1
        for entry in ctx.plan_history:
            assert isinstance(entry, PlanHistoryEntry)
            assert entry.plan_id.startswith("plan-attempt-")
            assert isinstance(entry.to_dict(), dict)


# ============================================================================
# L. PIPELINE INTEGRATION
# ============================================================================

class TestRecoveryPipelineIntegration:
    def test_standard_pipeline_with_recovery_engine_recovers_seamlessly(self):
        responses = {
            "search documentation and launch notepad": Result.fail(message="Connection reset timeout"),
        }
        flaky_router = FlakyMockRouter(failure_count=1, failure_message="Connection timeout")
        exec_engine = StandardExecutionEngine(router=flaky_router)

        pipeline = StandardPipeline(
            execution_engine=exec_engine,
            enable_recovery=True,
        )

        result = pipeline.process("what is quantum computing?")

        assert isinstance(result, PipelineResult)
        assert result.verification.verified is True
        assert result.recovery is not None
        assert result.recovery.attempt == 2
        assert "Recovered Output Data" in result.response


# ============================================================================
# 18. ARCHITECTURAL TEST: ZERO POLICY BYPASS ACROSS REPLANNING
# ============================================================================

class CustomStrictPolicyEngine(PolicyEngineInterface):
    """Strict policy engine that allows web search on attempt 1, but denies replanned alternative."""

    def __init__(self):
        self.evaluated_calls: List[ToolCall] = []

    def evaluate(self, tool_call: ToolCall, context: Optional[PolicyContext] = None) -> PolicyResult:
        self.evaluated_calls.append(tool_call)
        # Block any knowledge retrieval
        if tool_call.capability == "knowledge":
            return PolicyResult.deny(rule_id="strict_block", reason="Knowledge capability strictly prohibited by policy rule [strict_block].")
        return PolicyResult.allow(rule_id="allow_rule", reason="Allowed")



class TestArchitecturalReplanningSecurityBoundary:
    def test_replanned_tool_action_cannot_bypass_policy_or_orchestrator(self):
        """
        Prove that every candidate replan action MUST pass through ToolOrchestrator
        and PolicyEngine. A replanned action that policy denies is strictly blocked
        and never reaches executor.
        """
        strict_policy = CustomStrictPolicyEngine()
        orchestrator = ToolOrchestrator(policy_engine=strict_policy)

        # Step 1: Initial web search fails with insufficient evidence
        # Step 2: Replanning proposes knowledge retrieval fallback
        # Step 3: Knowledge retrieval MUST be intercepted and denied by strict_policy
        class MockRouter:
            def route(self, task: Task) -> Result:
                return orchestrator.execute_task(task)

        exec_engine = StandardExecutionEngine(router=MockRouter())
        verifier = StandardVerifier()


        # Custom planner that replans from web -> knowledge
        class KnowledgeFallbackPlanner(StandardRecoveryPlanner):
            def create_replan(self, context: RecoveryContext) -> Optional[Plan]:
                return Plan(
                    goal=context.original_goal,
                    steps=[
                        Task(id=1, type="knowledge", action="Retrieve Knowledge", tool="knowledge", parameters={"query": "test"}, status="pending")
                    ],
                    status="pending",
                )

        recovery_engine = StandardRecoveryEngine(planner=KnowledgeFallbackPlanner())

        initial_plan = Plan(
            goal="security test",
            steps=[
                Task(id=1, type="web", action="Web Search", tool="web", parameters={"query": "test"}, status="pending")
            ],
            status="pending",
        )

        # Simulate web search returning insufficient results
        call_count = 0
        def execute_fn(p: Plan) -> List[Result]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First attempt: web search fails
                p.status = "failed"
                return [Result.fail(message="insufficient evidence found")]
            else:
                # Replanned attempt: knowledge retrieval routed through orchestrator
                return exec_engine.execute(p)

        final_plan, results, ver, ctx = recovery_engine.recover(
            original_goal="security test",
            initial_plan=initial_plan,
            execute_fn=execute_fn,
            verify_fn=verifier.verify,
        )

        # Verify that PolicyEngine evaluated the replanned knowledge tool call
        evaluated_capabilities = [tc.capability for tc in strict_policy.evaluated_calls]
        assert "knowledge" in evaluated_capabilities

        # Verify that the replanned call was denied by policy
        assert ver.verified is False
        assert ctx.outcome == ExecutionOutcome.POLICY_BLOCKED
        assert any("Execution denied by policy" in r.message for r in results)


# ============================================================================
# L. METADATA & OBSERVABILITY
# ============================================================================

class TestRecoveryMetadataAndObservability:
    def test_recovery_metadata_and_observability_preservation(self):
        context = RecoveryContext(
            original_goal="research AI trends",
            current_plan=Plan(goal="research AI trends", steps=[], status="failed"),
            attempt=2,
            outcome=ExecutionOutcome.FAILURE,
            failure_classification=FailureClassification.TRANSIENT,
            failure_reason="timeout on gateway",
            replan_count=1,
            remaining_retry_budget=1,
            remaining_replan_budget=1,
            metadata={"source": "agent", "trace_id": "tr-1234"},
        )

        d = context.to_dict()
        assert d["original_goal"] == "research AI trends"
        assert d["attempt"] == 2
        assert d["outcome"] == "failure"
        assert d["failure_classification"] == "transient"
        assert d["failure_reason"] == "timeout on gateway"
        assert d["replan_count"] == 1
        assert d["remaining_retry_budget"] == 1
        assert d["remaining_replan_budget"] == 1
        assert d["metadata"]["trace_id"] == "tr-1234"



