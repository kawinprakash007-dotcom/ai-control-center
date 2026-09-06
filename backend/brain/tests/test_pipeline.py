import pytest
from dataclasses import replace
from datetime import datetime
from typing import Any, List, Optional

from core.interfaces.pipeline_interface import PipelineInterface
from core.interfaces.request_understanding_interface import RequestUnderstandingInterface
from core.interfaces.decision_engine_interface import DecisionEngineInterface
from core.interfaces.decision_planner_interface import DecisionPlannerInterface
from core.interfaces.execution_engine_interface import ExecutionEngineInterface
from core.interfaces.verification_interface import VerificationInterface
from core.interfaces.response_composer_interface import ResponseComposerInterface

from core.models.request import Request
from core.models.decision import Decision, CapabilityType, ExecutionMode
from core.models.plan import Plan
from core.models.task import Task
from core.models.result import Result
from core.models.verification import VerificationResult
from core.models.pipeline import PipelineResult

from brain.pipeline import StandardPipeline
from brain.request_understanding import StandardRequestUnderstanding, RequestUnderstandingError
from brain.decision_engine import StandardDecisionEngine
from brain.planning import StandardPlanner
from brain.execution import StandardExecutionEngine
from brain.verification import StandardVerifier
from brain.response import StandardResponseComposer


# ============================================================================
# TEST STUBS & SPY HELPERS
# ============================================================================

class StubRouter:
    """Fake router providing controlled results without invoking desktop tools or Ollama."""

    def __init__(self, default_result: Optional[Result] = None, responses: Optional[dict] = None):
        self.default_result = default_result or Result(
            success=True, message="Task completed successfully.", output="Stub Output"
        )
        self.responses = responses or {}
        self.routed_tasks: List[Task] = []

    def route(self, task: Task) -> Result:
        self.routed_tasks.append(task)
        return self.responses.get(task.id, self.default_result)


class SpyExecutionEngine(ExecutionEngineInterface):
    """Stub execution engine that tracks calls and applies execution statuses."""

    def __init__(self, results: Optional[List[Result]] = None):
        self.results = results if results is not None else [
            Result(success=True, message="Success", output="Executed Output")
        ]
        self.executed_plans: List[Plan] = []

    def execute(self, plan: Plan) -> List[Result]:
        self.executed_plans.append(plan)
        for idx, task in enumerate(plan.steps):
            task.status = "completed"
            task.result = "Executed Output"
        plan.status = "completed"
        return list(self.results)


class SpyStageTracker:
    """Helper to record the exact call sequence across all 6 pipeline stages."""

    def __init__(self):
        self.call_log: List[str] = []
        self.passed_arguments: dict = {}


# ============================================================================
# 1-5. CONTRACT TESTS
# ============================================================================

def test_pipeline_interface_compliance():
    pipeline = StandardPipeline(execution_engine=SpyExecutionEngine())
    assert isinstance(pipeline, PipelineInterface)
    assert hasattr(pipeline, "process")
    assert callable(pipeline.process)
    assert hasattr(pipeline, "run")
    assert callable(pipeline.run)


def test_process_returns_pipeline_result():
    pipeline = StandardPipeline(execution_engine=SpyExecutionEngine())
    result = pipeline.process("test query")
    assert isinstance(result, PipelineResult)
    assert isinstance(result.response, str)
    assert isinstance(result.request, Request)
    assert isinstance(result.decision, Decision)
    assert isinstance(result.plan, Plan)
    assert isinstance(result.results, list)
    assert isinstance(result.verification, VerificationResult)


def test_run_returns_response_string():
    pipeline = StandardPipeline(execution_engine=SpyExecutionEngine())
    response = pipeline.run("test query")
    assert isinstance(response, str)
    assert len(response) > 0


def test_run_delegates_to_process(monkeypatch):
    pipeline = StandardPipeline(execution_engine=SpyExecutionEngine())
    calls = []

    original_process = pipeline.process

    def mock_process(data):
        calls.append(data)
        return original_process(data)

    monkeypatch.setattr(pipeline, "process", mock_process)

    resp = pipeline.run("delegation test")
    assert len(calls) == 1
    assert calls[0] == "delegation test"
    assert isinstance(resp, str)


def test_invalid_input_raises_request_understanding_error():
    pipeline = StandardPipeline(execution_engine=SpyExecutionEngine())
    with pytest.raises(RequestUnderstandingError):
        pipeline.process(12345)  # Unsupported int input


# ============================================================================
# 6-11. LIFECYCLE TESTS & STAGE ORDERING & CHAINING
# ============================================================================

def test_exact_stage_ordering_and_parameter_chaining():
    tracker = SpyStageTracker()

    class TrackedUnderstanding(RequestUnderstandingInterface):
        def understand(self, input_data: Any) -> Request:
            tracker.call_log.append("understanding")
            tracker.passed_arguments["understanding_input"] = input_data
            return Request(
                id="req-1",
                original_text="text",
                normalized_text="text",
                session_id="s1",
                timestamp=datetime.now(),
            )

    class TrackedDecisionEngine(DecisionEngineInterface):
        def decide(self, request: Request) -> Decision:
            tracker.call_log.append("decision_engine")
            tracker.passed_arguments["decision_request"] = request
            return Decision(
                request_id=request.id,
                primary_goal="answer_chat",
                required_capabilities=[CapabilityType.CHAT],
                execution_mode=ExecutionMode.DIRECT,
                confidence=1.0,
                reasoning="chat",
            )

    class TrackedPlanner(DecisionPlannerInterface):
        def plan(self, decision: Decision) -> Plan:
            tracker.call_log.append("planner")
            tracker.passed_arguments["planner_decision"] = decision
            return Plan(
                goal=decision.primary_goal,
                steps=[Task(id=1, type="chat", action="Act", tool="chat")],
                status="pending",
            )

    class TrackedExecutionEngine(ExecutionEngineInterface):
        def execute(self, plan: Plan) -> List[Result]:
            tracker.call_log.append("execution_engine")
            tracker.passed_arguments["execution_plan"] = plan
            plan.status = "completed"
            plan.steps[0].status = "completed"
            return [Result(success=True, message="Done", output="Out")]

    class TrackedVerifier(VerificationInterface):
        def verify(self, plan: Plan, results: List[Result]) -> VerificationResult:
            tracker.call_log.append("verifier")
            tracker.passed_arguments["verifier_plan"] = plan
            tracker.passed_arguments["verifier_results"] = results
            return VerificationResult(
                verified=True, status="verified", confidence=1.0, reason="OK"
            )

    class TrackedComposer(ResponseComposerInterface):
        def compose(
            self,
            request: Request,
            decision: Decision,
            plan: Plan,
            results: List[Result],
            verification: VerificationResult,
        ) -> str:
            tracker.call_log.append("composer")
            tracker.passed_arguments["composer_args"] = (
                request,
                decision,
                plan,
                results,
                verification,
            )
            return "Composed Response"

    pipeline = StandardPipeline(
        understanding=TrackedUnderstanding(),
        decision_engine=TrackedDecisionEngine(),
        planner=TrackedPlanner(),
        execution_engine=TrackedExecutionEngine(),
        verifier=TrackedVerifier(),
        composer=TrackedComposer(),
    )

    pipeline_res = pipeline.process("input_payload")

    # 6. Exact stage ordering:
    assert tracker.call_log == [
        "understanding",
        "decision_engine",
        "planner",
        "execution_engine",
        "verifier",
        "composer",
    ]

    # Exactly once assertions:
    assert tracker.call_log.count("understanding") == 1
    assert tracker.call_log.count("decision_engine") == 1
    assert tracker.call_log.count("planner") == 1
    assert tracker.call_log.count("execution_engine") == 1
    assert tracker.call_log.count("verifier") == 1
    assert tracker.call_log.count("composer") == 1

    # 7-11. Chaining assertions:
    assert tracker.passed_arguments["understanding_input"] == "input_payload"
    assert tracker.passed_arguments["decision_request"] is pipeline_res.request
    assert tracker.passed_arguments["planner_decision"] is pipeline_res.decision
    assert tracker.passed_arguments["execution_plan"] is pipeline_res.plan
    assert tracker.passed_arguments["verifier_plan"] is pipeline_res.plan
    assert tracker.passed_arguments["verifier_results"] is pipeline_res.results
    assert tracker.passed_arguments["composer_args"] == (
        pipeline_res.request,
        pipeline_res.decision,
        pipeline_res.plan,
        pipeline_res.results,
        pipeline_res.verification,
    )
    assert pipeline_res.response == "Composed Response"


# ============================================================================
# 12-15. FULL PIPELINE & PRESERVATION TESTS
# ============================================================================

def test_full_successful_pipeline_with_stub_router():
    router = StubRouter(
        default_result=Result(success=True, message="Success", output="Retrieved Knowledge Excerpt")
    )
    exec_engine = StandardExecutionEngine(router=router)
    pipeline = StandardPipeline(execution_engine=exec_engine)

    res = pipeline.process("what is linux kernel?")

    assert res.response == "Retrieved Knowledge Excerpt"
    assert res.decision.primary_goal == "retrieve_knowledge"
    assert res.plan.status == "completed"
    assert res.verification.verified is True
    assert res.verification.status == "verified"
    assert len(res.results) == 1
    assert res.results[0].output == "Retrieved Knowledge Excerpt"


def test_multiple_results_preserved_in_pipeline():
    responses = {
        1: Result(success=True, message="Doc retrieved", output="Part 1"),
        2: Result(success=True, message="Notepad launched", output=None),
    }
    router = StubRouter(responses=responses)
    exec_engine = StandardExecutionEngine(router=router)
    pipeline = StandardPipeline(execution_engine=exec_engine)

    # Multi-step query
    res = pipeline.process("search documentation and launch notepad")

    assert res.decision.execution_mode == ExecutionMode.MULTI_STEP
    assert len(res.plan.steps) == 2
    assert len(res.results) == 2
    assert res.results[0].output == "Part 1"
    assert res.results[1].message == "Notepad launched"
    assert res.verification.verified is True
    assert "Part 1" in res.response
    assert "Notepad launched" in res.response


# ============================================================================
# 16-21. FAILURE BEHAVIOR & ERROR PROPAGATION
# ============================================================================

def test_request_understanding_error_propagates():
    pipeline = StandardPipeline(execution_engine=SpyExecutionEngine())
    with pytest.raises(RequestUnderstandingError):
        pipeline.process({"invalid": "missing text"})


def test_decision_engine_type_error_propagates():
    class BrokenDecisionEngine(DecisionEngineInterface):
        def decide(self, request: Request) -> Decision:
            raise TypeError("Simulated DecisionEngine TypeError")

    pipeline = StandardPipeline(
        decision_engine=BrokenDecisionEngine(),
        execution_engine=SpyExecutionEngine(),
    )
    with pytest.raises(TypeError) as excinfo:
        pipeline.process("valid text")
    assert "Simulated DecisionEngine TypeError" in str(excinfo.value)


def test_planner_value_error_propagates():
    class BrokenPlanner(DecisionPlannerInterface):
        def plan(self, decision: Decision) -> Plan:
            raise ValueError("Simulated Planner ValueError")

    pipeline = StandardPipeline(
        planner=BrokenPlanner(),
        execution_engine=SpyExecutionEngine(),
    )
    with pytest.raises(ValueError) as excinfo:
        pipeline.process("valid text")
    assert "Simulated Planner ValueError" in str(excinfo.value)


def test_execution_failure_result_flows_into_verifier():
    failing_router = StubRouter(
        default_result=Result(success=False, message="Tool failed to start", output=None)
    )
    exec_engine = StandardExecutionEngine(router=failing_router)
    pipeline = StandardPipeline(execution_engine=exec_engine)

    res = pipeline.process("open calculator")

    assert res.plan.status == "failed"
    assert res.results[0].success is False
    assert res.verification.verified is False
    assert res.verification.status == "failed"
    assert res.verification.failed_task_id == 1
    assert "Execution failed on task 1" in res.response
    assert "Tool failed to start" in res.response


def test_verification_exception_propagates():
    class BrokenVerifier(VerificationInterface):
        def verify(self, plan: Plan, results: List[Result]) -> VerificationResult:
            raise RuntimeError("Verification internal failure")

    pipeline = StandardPipeline(
        verifier=BrokenVerifier(),
        execution_engine=SpyExecutionEngine(),
    )
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.process("valid text")
    assert "Verification internal failure" in str(excinfo.value)


def test_response_composer_exception_propagates():
    class BrokenComposer(ResponseComposerInterface):
        def compose(self, req, dec, plan, results, ver) -> str:
            raise RuntimeError("Composer internal failure")

    pipeline = StandardPipeline(
        composer=BrokenComposer(),
        execution_engine=SpyExecutionEngine(),
    )
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.process("valid text")
    assert "Composer internal failure" in str(excinfo.value)


# ============================================================================
# 22-26. SPECIAL CASES (EMPTY, AMBIGUOUS, KNOWLEDGE, TOOL, UNSUPPORTED)
# ============================================================================

def test_empty_input_pipeline():
    router = StubRouter()
    exec_engine = StandardExecutionEngine(router=router)
    pipeline = StandardPipeline(execution_engine=exec_engine)

    res = pipeline.process("")
    assert res.decision.primary_goal == "prompt_user_input"
    assert res.plan.steps == []
    assert len(router.routed_tasks) == 0
    assert res.results == []
    assert res.verification.verified is True
    assert res.verification.reason == "Plan completed with no tasks."
    assert res.response == "Please provide an instruction or question."


def test_ambiguous_input_pipeline():
    router = StubRouter()
    exec_engine = StandardExecutionEngine(router=router)
    pipeline = StandardPipeline(execution_engine=exec_engine)

    res = pipeline.process("???")
    assert res.decision.primary_goal == "clarify_request"
    assert res.plan.steps == []
    assert len(router.routed_tasks) == 0
    assert res.results == []
    assert res.verification.verified is True
    assert res.verification.reason == "Plan completed with no tasks."
    assert "Please clarify your request" in res.response


def test_knowledge_flow_pipeline():
    knowledge_router = StubRouter(
        default_result=Result(
            success=True, message="Retrieved", output="Kernel architecture documentation"
        )
    )
    exec_engine = StandardExecutionEngine(router=knowledge_router)
    pipeline = StandardPipeline(execution_engine=exec_engine)

    res = pipeline.process("explain kernel architecture")
    assert res.decision.primary_goal == "retrieve_knowledge"
    assert res.plan.steps[0].type == "knowledge"
    assert res.plan.steps[0].tool == "knowledge"
    assert res.response == "Kernel architecture documentation"


def test_tool_flow_pipeline():
    tool_router = StubRouter(
        default_result=Result(success=True, message="Notepad launched", output=None)
    )
    exec_engine = StandardExecutionEngine(router=tool_router)
    pipeline = StandardPipeline(execution_engine=exec_engine)

    res = pipeline.process("open notepad")
    assert res.decision.primary_goal == "execute_tool"
    assert res.plan.steps[0].type == "tool"
    assert res.plan.steps[0].tool == "notepad"
    assert res.response == "Notepad launched"


def test_unsupported_capability_as_failed_result():
    # Router returns failure for unknown tool
    fail_router = StubRouter(
        default_result=Result(success=False, message="No capability found for 'unknown_tool'")
    )
    exec_engine = StandardExecutionEngine(router=fail_router)
    pipeline = StandardPipeline(execution_engine=exec_engine)

    res = pipeline.process("open unknown_tool")
    assert res.verification.verified is False
    assert res.verification.status == "failed"
    assert "No capability found for 'unknown_tool'" in res.response


# ============================================================================
# 27-30. ARCHITECTURAL ISOLATION & IMMUTABILITY
# ============================================================================

def test_no_v1_agent_dependency():
    import brain.pipeline.standard_pipeline as sp_mod
    assert not hasattr(sp_mod, "Agent")
    assert not hasattr(sp_mod, "Reasoner")
    assert not hasattr(sp_mod, "GoalManager")
    assert not hasattr(sp_mod, "Reflection")
    assert not hasattr(sp_mod, "TaskManager")


def test_no_ollama_dependency():
    import brain.pipeline.standard_pipeline as sp_mod
    assert not hasattr(sp_mod, "ask_ollama")
    assert not hasattr(sp_mod, "OllamaClient")


def test_dependency_injection_all_six_components():
    fake_und = StandardRequestUnderstanding()
    fake_dec = StandardDecisionEngine()
    fake_pln = StandardPlanner()
    fake_exe = SpyExecutionEngine()
    fake_ver = StandardVerifier()
    fake_cmp = StandardResponseComposer()

    pipeline = StandardPipeline(
        understanding=fake_und,
        decision_engine=fake_dec,
        planner=fake_pln,
        execution_engine=fake_exe,
        verifier=fake_ver,
        composer=fake_cmp,
    )

    assert pipeline.understanding is fake_und
    assert pipeline.decision_engine is fake_dec
    assert pipeline.planner is fake_pln
    assert pipeline.execution_engine is fake_exe
    assert pipeline.verifier is fake_ver
    assert pipeline.composer is fake_cmp


def test_pipeline_result_immutability():
    pipeline = StandardPipeline(execution_engine=SpyExecutionEngine())
    res = pipeline.process("test query")

    with pytest.raises(Exception):
        res.response = "Mutated"  # type: ignore

    with pytest.raises(Exception):
        res.request = None  # type: ignore


# ============================================================================
# 31-34. INTEGRITY ASSERTIONS
# ============================================================================

def test_stage_integrity_no_unexpected_mutations():
    pipeline = StandardPipeline(execution_engine=SpyExecutionEngine())
    res = pipeline.process("integrity test")

    # Request, Decision, VerificationResult are frozen dataclasses
    assert res.request.original_text == "integrity test"
    assert res.decision.primary_goal is not None
    assert res.verification.verified is True

    # Plan execution mutations are strictly limited to plan.status, task.status, task.result
    for task in res.plan.steps:
        assert task.status == "completed"
        assert task.result is not None
        # Action, type, tool were preserved
        assert task.tool == "chat"
