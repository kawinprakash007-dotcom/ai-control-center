import pytest
from datetime import datetime
from typing import List

from core.interfaces.response_composer_interface import ResponseComposerInterface
from core.models.request import Request
from core.models.decision import Decision, CapabilityType, ExecutionMode
from core.models.plan import Plan
from core.models.task import Task
from core.models.result import Result
from core.models.verification import VerificationResult
from brain.response import StandardResponseComposer


def make_request(text: str = "test query", parameters: dict = None, constraints: dict = None) -> Request:
    return Request(
        id="req-123",
        original_text=text,
        normalized_text=text.lower().strip(),
        session_id="session-1",
        timestamp=datetime.now(),
        source="chat",
        parameters=parameters or {},
        constraints=constraints or {},
    )


def make_decision(
    primary_goal: str = "answer_chat",
    capabilities: List[CapabilityType] = None,
    mode: ExecutionMode = ExecutionMode.DIRECT,
    routing_hints: dict = None,
) -> Decision:
    return Decision(
        request_id="req-123",
        primary_goal=primary_goal,
        required_capabilities=capabilities or [CapabilityType.CHAT],
        execution_mode=mode,
        confidence=0.95,
        reasoning="Test decision reasoning.",
        routing_hints=routing_hints or {},
    )


def make_plan(steps: List[Task] = None, goal: str = "test_goal") -> Plan:
    return Plan(
        goal=goal,
        steps=steps or [],
        status="completed",
        confidence=0.95,
    )


def make_task(task_id: int = 1, tool: str = "chat", status: str = "completed") -> Task:
    return Task(
        id=task_id,
        type="chat",
        action=f"Execute {tool}",
        tool=tool,
        status=status,
    )


# ============================================================================
# 1. INTERFACE COMPLIANCE
# ============================================================================

def test_interface_compliance():
    composer = StandardResponseComposer()
    assert isinstance(composer, ResponseComposerInterface)
    assert hasattr(composer, "compose")
    assert callable(composer.compose)


# ============================================================================
# 2. TYPE SAFETY
# ============================================================================

def test_type_safety_invalid_inputs():
    composer = StandardResponseComposer()
    req = make_request()
    dec = make_decision()
    plan = make_plan()
    results = [Result(success=True, message="OK")]
    ver = VerificationResult(verified=True, status="verified", confidence=1.0, reason="OK")

    with pytest.raises(TypeError):
        composer.compose("not a request", dec, plan, results, ver)  # type: ignore

    with pytest.raises(TypeError):
        composer.compose(req, "not a decision", plan, results, ver)  # type: ignore

    with pytest.raises(TypeError):
        composer.compose(req, dec, "not a plan", results, ver)  # type: ignore

    with pytest.raises(TypeError):
        composer.compose(req, dec, plan, "not a list", ver)  # type: ignore

    with pytest.raises(TypeError):
        composer.compose(req, dec, plan, results, "not a verification")  # type: ignore


# ============================================================================
# 3. EMPTY INPUT / PROMPT USER INPUT
# ============================================================================

def test_empty_input_prompt_user_input():
    composer = StandardResponseComposer()
    req = make_request("", parameters={"is_empty": True})
    dec = make_decision(primary_goal="prompt_user_input")
    plan = make_plan()
    results = []
    ver = VerificationResult(verified=True, status="verified", confidence=1.0, reason="Plan completed with no tasks.")

    response = composer.compose(req, dec, plan, results, ver)
    assert response == "Please provide an instruction or question."


# ============================================================================
# 4. CLARIFICATION REQUEST
# ============================================================================

def test_clarification_request():
    composer = StandardResponseComposer()
    req = make_request("run it", parameters={"is_ambiguous": True, "ambiguity_reason": "Missing target tool"})
    dec = make_decision(primary_goal="clarify_request", routing_hints={"ambiguity_reason": "Missing target tool"})
    plan = make_plan()
    results = []
    ver = VerificationResult(verified=True, status="verified", confidence=1.0, reason="Plan completed with no tasks.")

    response = composer.compose(req, dec, plan, results, ver)
    assert "Please clarify your request" in response
    assert "Missing target tool" in response


# ============================================================================
# 5. SUCCESSFUL SINGLE RESULT WITH OUTPUT
# ============================================================================

def test_successful_single_result_with_output():
    composer = StandardResponseComposer()
    req = make_request("what is linux?")
    dec = make_decision(primary_goal="retrieve_knowledge")
    plan = make_plan([make_task(1, "knowledge")])
    results = [Result(success=True, message="Retrieved", output="Linux is an open-source monolithic kernel.")]
    ver = VerificationResult(verified=True, status="verified", confidence=1.0, reason="All planned tasks completed successfully.")

    response = composer.compose(req, dec, plan, results, ver)
    assert response == "Linux is an open-source monolithic kernel."


# ============================================================================
# 6. SUCCESSFUL RESULT WITH NONE OUTPUT (USES MESSAGE)
# ============================================================================

def test_successful_result_none_output():
    composer = StandardResponseComposer()
    req = make_request("open notepad")
    dec = make_decision(primary_goal="execute_tool")
    plan = make_plan([make_task(1, "notepad")])
    results = [Result(success=True, message="Notepad opened successfully.", output=None)]
    ver = VerificationResult(verified=True, status="verified", confidence=1.0, reason="All planned tasks completed successfully.")

    response = composer.compose(req, dec, plan, results, ver)
    assert response == "Notepad opened successfully."


# ============================================================================
# 7. SUCCESSFUL RESULT WITH EMPTY OUTPUT (USES MESSAGE)
# ============================================================================

def test_successful_result_empty_output():
    composer = StandardResponseComposer()
    req = make_request("open calculator")
    dec = make_decision(primary_goal="execute_tool")
    plan = make_plan([make_task(1, "calculator")])
    results = [Result(success=True, message="Calculator launched.", output="   ")]
    ver = VerificationResult(verified=True, status="verified", confidence=1.0, reason="All planned tasks completed successfully.")

    response = composer.compose(req, dec, plan, results, ver)
    assert response == "Calculator launched."


# ============================================================================
# 8. SUCCESSFUL MULTIPLE RESULTS
# ============================================================================

def test_successful_multiple_results_ordered_separation():
    composer = StandardResponseComposer()
    req = make_request("retrieve and launch")
    dec = make_decision(primary_goal="execute_workflow", mode=ExecutionMode.MULTI_STEP)
    plan = make_plan([make_task(1, "knowledge"), make_task(2, "notepad")])
    results = [
        Result(success=True, message="Retrieved", output="Kernel docs excerpt."),
        Result(success=True, message="Notepad launched.", output=None),
    ]
    ver = VerificationResult(verified=True, status="verified", confidence=1.0, reason="All planned tasks completed successfully.")

    response = composer.compose(req, dec, plan, results, ver)
    assert response == "Kernel docs excerpt.\nNotepad launched."


# ============================================================================
# 9 & 10. FAILURE RESPONSE & FAILED TASK ID INCLUDED
# ============================================================================

def test_failure_response_with_task_id():
    composer = StandardResponseComposer()
    req = make_request("open calculator")
    dec = make_decision(primary_goal="execute_tool")
    t1 = make_task(1, "calculator", status="failed")
    plan = make_plan([t1])
    results = [Result(success=False, message="Calculator binary not found.", output=None)]
    ver = VerificationResult(
        verified=False,
        status="failed",
        confidence=1.0,
        reason="Execution failed on task 1: Calculator binary not found.",
        failed_task_id=1,
    )

    response = composer.compose(req, dec, plan, results, ver)
    assert "Execution failed on task 1" in response
    assert "Calculator binary not found" in response


# ============================================================================
# 11. UNVERIFIED RESPONSE
# ============================================================================

def test_unverified_response():
    composer = StandardResponseComposer()
    req = make_request("partial task")
    dec = make_decision(primary_goal="execute_tool")
    plan = make_plan([make_task(1, "notepad")])
    results = []
    ver = VerificationResult(
        verified=False,
        status="unverified",
        confidence=0.5,
        reason="Execution state is incomplete or inconsistent.",
    )

    response = composer.compose(req, dec, plan, results, ver)
    assert "could not be fully verified" in response
    assert "incomplete or inconsistent" in response


# ============================================================================
# 12. KNOWLEDGE RESULT PRESERVATION
# ============================================================================

def test_knowledge_result_exact_preservation():
    composer = StandardResponseComposer()
    knowledge_text = "### Architecture Overview\n- Layer 1: Brain\n- Layer 2: Tools"
    req = make_request("explain architecture")
    dec = make_decision(primary_goal="retrieve_knowledge")
    plan = make_plan([make_task(1, "knowledge")])
    results = [Result(success=True, message="OK", output=knowledge_text)]
    ver = VerificationResult(verified=True, status="verified", confidence=1.0, reason="All planned tasks completed successfully.")

    response = composer.compose(req, dec, plan, results, ver)
    assert response == knowledge_text


# ============================================================================
# 13. ARBITRARY OUTPUT STRING HANDLING
# ============================================================================

def test_arbitrary_output_string_handling():
    composer = StandardResponseComposer()
    complex_text = "JSON: {\"key\": \"value\", \"list\": [1, 2, 3]}\nSymbols: !@#$%^&*()_+"
    req = make_request("generate payload")
    dec = make_decision()
    plan = make_plan([make_task(1)])
    results = [Result(success=True, message="Done", output=complex_text)]
    ver = VerificationResult(verified=True, status="verified", confidence=1.0, reason="All planned tasks completed successfully.")

    response = composer.compose(req, dec, plan, results, ver)
    assert response == complex_text


# ============================================================================
# 14. NO LLM / OLLAMA DEPENDENCY
# ============================================================================

def test_no_llm_or_ollama_dependency():
    import brain.response.standard_response_composer as src_mod
    assert not hasattr(src_mod, "ask_ollama")
    assert not hasattr(src_mod, "OllamaClient")
    assert not hasattr(src_mod, "Router")


# ============================================================================
# 15. INPUT MODELS REMAIN UNCHANGED
# ============================================================================

def test_input_models_remain_unchanged():
    composer = StandardResponseComposer()
    req = make_request("do something", parameters={"foo": "bar"})
    dec = make_decision(routing_hints={"hint": 123})
    plan = make_plan([make_task(1)])
    results = [Result(success=True, message="OK", output="payload")]
    ver = VerificationResult(verified=True, status="verified", confidence=1.0, reason="OK")

    composer.compose(req, dec, plan, results, ver)

    assert req.original_text == "do something"
    assert req.parameters == {"foo": "bar"}
    assert dec.routing_hints == {"hint": 123}
    assert plan.goal == "test_goal"
    assert results[0].output == "payload"
    assert ver.verified is True
