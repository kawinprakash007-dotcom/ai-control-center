import pytest
from typing import List

from core.interfaces.verification_interface import VerificationInterface
from core.models.plan import Plan
from core.models.task import Task
from core.models.result import Result
from core.models.verification import VerificationResult
from brain.verification import StandardVerifier


def make_task(
    task_id: int = 1,
    tool: str = "chat",
    task_type: str = "chat",
    status: str = "completed",
    result: str = "Done",
) -> Task:
    return Task(
        id=task_id,
        type=task_type,
        action=f"Execute {tool}",
        tool=tool,
        parameters={},
        status=status,
        result=result,
    )


def make_plan(
    steps: List[Task] = None,
    goal: str = "test_goal",
    status: str = "completed",
    confidence: float = 0.95,
) -> Plan:
    return Plan(
        goal=goal,
        steps=steps or [],
        status=status,
        confidence=confidence,
    )


# ============================================================================
# 1. INTERFACE COMPLIANCE
# ============================================================================

def test_interface_compliance():
    verifier = StandardVerifier()
    assert isinstance(verifier, VerificationInterface)
    assert hasattr(verifier, "verify")
    assert callable(verifier.verify)


# ============================================================================
# 2 & 3. TYPE SAFETY
# ============================================================================

def test_invalid_plan_type_raises_type_error():
    verifier = StandardVerifier()
    with pytest.raises(TypeError) as excinfo:
        verifier.verify("not a plan", [])  # type: ignore
    assert "expects a Plan instance" in str(excinfo.value)

    with pytest.raises(TypeError):
        verifier.verify({"steps": []}, [])  # type: ignore

    with pytest.raises(TypeError):
        verifier.verify(None, [])  # type: ignore


def test_invalid_results_type_raises_type_error():
    verifier = StandardVerifier()
    plan = make_plan()
    with pytest.raises(TypeError) as excinfo:
        verifier.verify(plan, "not a list")  # type: ignore
    assert "expects a list for results" in str(excinfo.value)

    with pytest.raises(TypeError):
        verifier.verify(plan, None)  # type: ignore


# ============================================================================
# 4. EMPTY PLAN
# ============================================================================

def test_empty_plan_verification():
    verifier = StandardVerifier()
    plan = make_plan(steps=[], status="completed")
    res = verifier.verify(plan, [])

    assert isinstance(res, VerificationResult)
    assert res.verified is True
    assert res.status == "verified"
    assert res.confidence == 1.0
    assert res.reason == "Plan completed with no tasks."
    assert res.failed_task_id is None


# ============================================================================
# 5. SUCCESSFUL SINGLE-TASK PLAN
# ============================================================================

def test_successful_single_task_plan():
    verifier = StandardVerifier()
    task = make_task(task_id=1, status="completed", result="Answer text")
    plan = make_plan(steps=[task], status="completed")
    results = [Result(success=True, message="Success", output="Answer text")]

    res = verifier.verify(plan, results)

    assert res.verified is True
    assert res.status == "verified"
    assert res.confidence == 1.0
    assert res.reason == "All planned tasks completed successfully."
    assert res.failed_task_id is None


# ============================================================================
# 6. SUCCESSFUL MULTI-TASK PLAN
# ============================================================================

def test_successful_multi_task_plan():
    verifier = StandardVerifier()
    t1 = make_task(task_id=1, tool="knowledge", task_type="knowledge", status="completed", result="Doc")
    t2 = make_task(task_id=2, tool="notepad", task_type="tool", status="completed", result="Opened")
    plan = make_plan(steps=[t1, t2], status="completed")
    results = [
        Result(success=True, message="Retrieved", output="Doc"),
        Result(success=True, message="Launched", output="Opened"),
    ]

    res = verifier.verify(plan, results)

    assert res.verified is True
    assert res.status == "verified"
    assert res.confidence == 1.0
    assert res.failed_task_id is None


# ============================================================================
# 7 & 11. FAILED TASK & FAILED TASK IDENTIFICATION
# ============================================================================

def test_failed_task_identification():
    verifier = StandardVerifier()
    t1 = make_task(task_id=1, status="completed", result="OK")
    t2 = make_task(task_id=2, status="failed", result="Calculator failed")
    t3 = make_task(task_id=3, status="skipped")
    plan = make_plan(steps=[t1, t2, t3], status="failed")
    results = [
        Result(success=True, message="OK", output="OK"),
        Result(success=False, message="Calculator failed", output=None),
    ]

    res = verifier.verify(plan, results)

    assert res.verified is False
    assert res.status == "failed"
    assert res.confidence == 1.0
    assert res.failed_task_id == 2
    assert "task 2" in res.reason.lower() or "calculator failed" in res.reason.lower()


# ============================================================================
# 8. FAILED RESULT
# ============================================================================

def test_failed_result_triggers_verification_failure():
    verifier = StandardVerifier()
    t1 = make_task(task_id=10, status="completed", result="OK")
    plan = make_plan(steps=[t1], status="completed")
    results = [Result(success=False, message="External error occurred", output=None)]

    res = verifier.verify(plan, results)

    assert res.verified is False
    assert res.status == "failed"
    assert res.failed_task_id == 10
    assert "External error occurred" in res.reason


# ============================================================================
# 9. SKIPPED TASK
# ============================================================================

def test_skipped_task_triggers_verification_failure():
    verifier = StandardVerifier()
    t1 = make_task(task_id=1, status="completed", result="OK")
    t2 = make_task(task_id=2, status="skipped")
    # Even if plan.status wasn't explicitly failed, skipped task fails verification
    plan = make_plan(steps=[t1, t2], status="completed")
    results = [Result(success=True, message="OK", output="OK")]

    res = verifier.verify(plan, results)

    assert res.verified is False
    assert res.status == "failed"
    assert res.failed_task_id == 2
    assert "skipped" in res.reason.lower()


# ============================================================================
# 10. INCONSISTENT / INCOMPLETE EXECUTION STATE
# ============================================================================

def test_inconsistent_execution_state():
    verifier = StandardVerifier()
    t1 = make_task(task_id=1, status="pending")  # Still pending!
    plan = make_plan(steps=[t1], status="pending")
    results = []

    res = verifier.verify(plan, results)

    assert res.verified is False
    assert res.status == "unverified"
    assert res.confidence <= 0.5
    assert "incomplete or inconsistent" in res.reason


def test_inconsistent_result_count():
    verifier = StandardVerifier()
    t1 = make_task(task_id=1, status="completed")
    t2 = make_task(task_id=2, status="completed")
    plan = make_plan(steps=[t1, t2], status="completed")
    # Only 1 result provided for 2 completed tasks
    results = [Result(success=True, message="OK", output="OK")]

    res = verifier.verify(plan, results)

    assert res.verified is False
    assert res.status == "unverified"
    assert res.confidence <= 0.5


# ============================================================================
# 12. IMMUTABILITY / NO MUTATION OF PLAN, TASK, RESULT
# ============================================================================

def test_plan_task_result_immutability():
    verifier = StandardVerifier()
    t1 = make_task(task_id=1, status="completed", result="output_1")
    plan = make_plan(steps=[t1], goal="immutable_goal", status="completed", confidence=0.88)
    r1 = Result(success=True, message="msg_1", output="output_1")
    results = [r1]

    verifier.verify(plan, results)

    assert plan.goal == "immutable_goal"
    assert plan.status == "completed"
    assert plan.confidence == 0.88
    assert t1.id == 1
    assert t1.status == "completed"
    assert t1.result == "output_1"
    assert r1.success is True
    assert r1.message == "msg_1"
    assert r1.output == "output_1"


# ============================================================================
# 13. CONFIDENCE BOUNDS & VERIFICATION RESULT VALIDATION
# ============================================================================

def test_verification_result_validation():
    # Valid
    v = VerificationResult(verified=True, status="verified", confidence=1.0, reason="ok")
    assert v.verified is True

    # Invalid status
    with pytest.raises(ValueError):
        VerificationResult(verified=True, status="unknown_status", confidence=1.0, reason="ok")

    # Invalid confidence range
    with pytest.raises(ValueError):
        VerificationResult(verified=True, status="verified", confidence=1.5, reason="ok")
    with pytest.raises(ValueError):
        VerificationResult(verified=True, status="verified", confidence=-0.1, reason="ok")

    # Inconsistent verified flag with status
    with pytest.raises(ValueError):
        VerificationResult(verified=False, status="verified", confidence=1.0, reason="ok")
    with pytest.raises(ValueError):
        VerificationResult(verified=True, status="failed", confidence=1.0, reason="ok")
    with pytest.raises(ValueError):
        VerificationResult(verified=True, status="unverified", confidence=0.5, reason="ok")


# ============================================================================
# 14. KNOWLEDGE-SHAPED SUCCESSFUL TASK
# ============================================================================

def test_knowledge_shaped_task_verification():
    verifier = StandardVerifier()
    task = make_task(
        task_id=1,
        tool="knowledge",
        task_type="knowledge",
        status="completed",
        result="Kernel documentation summary text",
    )
    plan = make_plan(steps=[task], goal="retrieve_knowledge", status="completed")
    results = [Result(success=True, message="Retrieved", output="Kernel documentation summary text")]

    res = verifier.verify(plan, results)

    assert res.verified is True
    assert res.status == "verified"
    assert res.confidence == 1.0
    assert res.reason == "All planned tasks completed successfully."


# ============================================================================
# 15. NO EXTERNAL DEPENDENCIES
# ============================================================================

def test_no_external_dependencies():
    import brain.verification.standard_verifier as sv_mod
    assert not hasattr(sv_mod, "ask_ollama")
    assert not hasattr(sv_mod, "Router")
    assert not hasattr(sv_mod, "Chroma")
    assert not hasattr(sv_mod, "TOOL_REGISTRY")
