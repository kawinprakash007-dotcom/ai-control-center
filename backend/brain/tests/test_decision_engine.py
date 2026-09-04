import pytest
from datetime import datetime
import dataclasses

from core.interfaces.decision_engine_interface import DecisionEngineInterface
from core.models.request import Request
from core.models.decision import (
    Decision,
    CapabilityType,
    ExecutionMode,
)
from brain.decision_engine import StandardDecisionEngine


@pytest.fixture
def engine():
    return StandardDecisionEngine()


def make_request(
    text: str,
    parameters: dict = None,
    constraints: dict = None,
    req_id: str = "req-001",
) -> Request:
    return Request(
        id=req_id,
        original_text=text,
        normalized_text=text,
        session_id="test-session",
        timestamp=datetime.now(),
        parameters=parameters or {},
        constraints=constraints or {},
    )


# ============================================================================
# 1. INTERFACE COMPLIANCE
# ============================================================================

def test_interface_compliance(engine):
    """1. StandardDecisionEngine must implement DecisionEngineInterface."""
    assert isinstance(engine, DecisionEngineInterface)
    assert hasattr(engine, "decide")
    assert callable(engine.decide)


# ============================================================================
# 2. REQUEST -> DECISION CORRELATION
# ============================================================================

def test_request_decision_correlation(engine):
    """2. Decision preserves request.id as Decision.request_id."""
    req = make_request("Hello", req_id="req-custom-999")
    decision = engine.decide(req)
    assert decision.request_id == "req-custom-999"


# ============================================================================
# 3. EMPTY REQUEST
# ============================================================================

def test_empty_request_decision(engine):
    """3. Empty input triggers prompt_user_input, CHAT, and DIRECT."""
    req = make_request("", parameters={"is_empty": True, "valid": False})
    decision = engine.decide(req)
    assert decision.primary_goal == "prompt_user_input"
    assert decision.required_capabilities == [CapabilityType.CHAT]
    assert decision.execution_mode == ExecutionMode.DIRECT
    assert decision.confidence == 1.0
    assert decision.routing_hints.get("is_empty") is True


# ============================================================================
# 4. AMBIGUOUS REQUEST
# ============================================================================

def test_ambiguous_request_decision(engine):
    """4. Ambiguous input triggers clarify_request, CHAT, DIRECT, and propagates ambiguity_reason."""
    req = make_request(
        "???",
        parameters={
            "is_ambiguous": True,
            "ambiguity_reason": "Input contains only punctuation",
            "clarification_needed": True,
        },
    )
    decision = engine.decide(req)
    assert decision.primary_goal == "clarify_request"
    assert decision.required_capabilities == [CapabilityType.CHAT]
    assert decision.execution_mode == ExecutionMode.DIRECT
    assert decision.confidence == 0.95
    assert "punctuation" in decision.reasoning.lower()
    assert decision.routing_hints["ambiguity_reason"] == "Input contains only punctuation"


# ============================================================================
# 5. KNOWLEDGE REQUEST
# ============================================================================

def test_knowledge_request_decision(engine):
    """5. Knowledge request triggers retrieve_knowledge, KNOWLEDGE, SINGLE_STEP."""
    req = make_request(
        "search for kernel architecture",
        parameters={"query": "kernel architecture"},
    )
    decision = engine.decide(req)
    assert decision.primary_goal == "retrieve_knowledge"
    assert decision.required_capabilities == [CapabilityType.KNOWLEDGE]
    assert decision.execution_mode == ExecutionMode.SINGLE_STEP
    assert decision.confidence >= 0.9
    assert decision.routing_hints["query"] == "kernel architecture"


# ============================================================================
# 6. TOOL REQUEST
# ============================================================================

def test_tool_request_decision(engine):
    """6. Tool request triggers execute_tool, TOOL, SINGLE_STEP, and provides tool_hint."""
    req = make_request("launch calculator")
    decision = engine.decide(req)
    assert decision.primary_goal == "execute_tool"
    assert decision.required_capabilities == [CapabilityType.TOOL]
    assert decision.execution_mode == ExecutionMode.SINGLE_STEP
    assert decision.confidence >= 0.9
    assert decision.routing_hints.get("tool_hint") == "calculator"


# ============================================================================
# 7. MEMORY REQUEST
# ============================================================================

def test_memory_request_decision(engine):
    """7. Memory request triggers manage_memory, MEMORY, SINGLE_STEP."""
    req = make_request("remember that my preferred IDE is VSCode")
    decision = engine.decide(req)
    assert decision.primary_goal == "manage_memory"
    assert decision.required_capabilities == [CapabilityType.MEMORY]
    assert decision.execution_mode == ExecutionMode.SINGLE_STEP
    assert decision.confidence >= 0.9


# ============================================================================
# 8. GENERAL CHAT
# ============================================================================

def test_general_chat_decision(engine):
    """8. General chat triggers answer_chat, CHAT, DIRECT."""
    req = make_request("good morning, how are you?")
    decision = engine.decide(req)
    assert decision.primary_goal == "answer_chat"
    assert decision.required_capabilities == [CapabilityType.CHAT]
    assert decision.execution_mode == ExecutionMode.DIRECT
    assert 0.0 <= decision.confidence <= 1.0


# ============================================================================
# 9. MULTI-STEP REQUEST
# ============================================================================

def test_multi_step_request_decision(engine):
    """9. Composite request triggers execute_workflow, MULTI_STEP, with multiple capabilities."""
    req = make_request(
        "search documentation for install guide and then open notepad",
        parameters={"query": "install guide"},
    )
    decision = engine.decide(req)
    assert decision.primary_goal == "execute_workflow"
    assert decision.execution_mode == ExecutionMode.MULTI_STEP
    assert CapabilityType.KNOWLEDGE in decision.required_capabilities
    assert CapabilityType.TOOL in decision.required_capabilities


# ============================================================================
# 10. ROUTING HINT PROPAGATION
# ============================================================================

def test_routing_hints_propagation(engine):
    """10. Parameters like path, query, collection propagate into routing_hints."""
    req = make_request(
        "inspect document",
        parameters={
            "path": "C:\\logs\\system.log",
            "query": "error code 500",
            "collection": "server_logs",
        },
    )
    decision = engine.decide(req)
    assert decision.routing_hints["path"] == "C:\\logs\\system.log"
    assert decision.routing_hints["query"] == "error code 500"
    assert decision.routing_hints["collection"] == "server_logs"


# ============================================================================
# 11. CONSTRAINTS PROPAGATION
# ============================================================================

def test_constraints_propagation(engine):
    """11. Constraints like max_results, timeout_seconds propagate into routing_hints."""
    req = make_request(
        "search docs",
        constraints={
            "max_results": 10,
            "timeout_seconds": 15.0,
            "privacy_level": "local_only",
            "confirmation_required": True,
        },
    )
    decision = engine.decide(req)
    assert decision.routing_hints["max_results"] == 10
    assert decision.routing_hints["timeout_seconds"] == 15.0
    assert decision.routing_hints["privacy_level"] == "local_only"
    assert decision.routing_hints["confirmation_required"] is True


# ============================================================================
# 12. DETERMINISTIC OUTPUT
# ============================================================================

def test_deterministic_output(engine):
    """12. Calling decide produces deterministic outputs given deterministic inputs."""
    req = make_request("find operating system notes", parameters={"query": "operating system notes"})
    decision = engine.decide(req)
    assert decision.primary_goal == "retrieve_knowledge"
    assert decision.confidence == 0.95
    assert decision.execution_mode == ExecutionMode.SINGLE_STEP


# ============================================================================
# 13. DECISION IMMUTABILITY
# ============================================================================

def test_decision_immutability(engine):
    """13. Returned Decision is frozen and immutable."""
    req = make_request("open chrome")
    decision = engine.decide(req)
    assert isinstance(decision, Decision)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.primary_goal = "modified_goal"  # type: ignore
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.confidence = 0.5  # type: ignore


# ============================================================================
# 14. TYPE ERROR FOR NON-REQUEST INPUT
# ============================================================================

def test_type_error_for_non_request_input(engine):
    """14. Passing a non-Request object must raise TypeError."""
    with pytest.raises(TypeError) as excinfo:
        engine.decide("just a string")  # type: ignore
    assert "expects a Request instance" in str(excinfo.value)

    with pytest.raises(TypeError):
        engine.decide({"text": "a dict"})  # type: ignore

    with pytest.raises(TypeError):
        engine.decide(None)  # type: ignore


# ============================================================================
# 15. CONFIDENCE IN BOUNDED RANGE [0.0, 1.0]
# ============================================================================

def test_confidence_bounded_range(engine):
    """15. Decision confidence is always bounded within [0.0, 1.0]."""
    test_inputs = [
        make_request(""),
        make_request("???", parameters={"is_ambiguous": True}),
        make_request("search for linux kernel"),
        make_request("open notepad"),
        make_request("remember my key"),
        make_request("just chatting"),
    ]
    for req in test_inputs:
        decision = engine.decide(req)
        assert 0.0 <= decision.confidence <= 1.0


# ============================================================================
# 16. REQUIRED CAPABILITIES ARE CORRECT
# ============================================================================

def test_required_capabilities_correctness(engine):
    """16. Required capabilities accurately reflect request type."""
    chat_dec = engine.decide(make_request("hello"))
    assert chat_dec.required_capabilities == [CapabilityType.CHAT]

    know_dec = engine.decide(make_request("search for compiler docs", parameters={"query": "compiler"}))
    assert know_dec.required_capabilities == [CapabilityType.KNOWLEDGE]

    tool_dec = engine.decide(make_request("launch notepad"))
    assert tool_dec.required_capabilities == [CapabilityType.TOOL]

    mem_dec = engine.decide(make_request("store preference dark mode"))
    assert mem_dec.required_capabilities == [CapabilityType.MEMORY]


# ============================================================================
# 17. EXECUTION MODE IS CORRECT
# ============================================================================

def test_execution_mode_correctness(engine):
    """17. Execution mode maps to DIRECT, SINGLE_STEP, or MULTI_STEP properly."""
    direct_dec = engine.decide(make_request("hello friend"))
    assert direct_dec.execution_mode == ExecutionMode.DIRECT

    single_dec = engine.decide(make_request("search docs", parameters={"query": "docs"}))
    assert single_dec.execution_mode == ExecutionMode.SINGLE_STEP

    multi_dec = engine.decide(make_request("search docs and then open notepad", parameters={"query": "docs"}))
    assert multi_dec.execution_mode == ExecutionMode.MULTI_STEP


# ============================================================================
# 18. REQUEST PARAMETERS NOT MUTATED
# ============================================================================

def test_request_parameters_not_mutated(engine):
    """18. Existing Request parameters dictionary is not mutated."""
    params = {"query": "initial_query", "custom_key": "val"}
    req = make_request("search docs", parameters=params)
    snapshot = dict(req.parameters)

    _ = engine.decide(req)
    assert req.parameters == snapshot


# ============================================================================
# 19. REQUEST CONSTRAINTS NOT MUTATED
# ============================================================================

def test_request_constraints_not_mutated(engine):
    """19. Existing Request constraints dictionary is not mutated."""
    constraints = {"max_results": 5, "timeout_seconds": 10.0}
    req = make_request("search docs", constraints=constraints)
    snapshot = dict(req.constraints)

    _ = engine.decide(req)
    assert req.constraints == snapshot


# ============================================================================
# 20. MULTIPLE CALLS PRODUCE EQUIVALENT DECISIONS
# ============================================================================

def test_multiple_calls_produce_equivalent_decisions(engine):
    """20. Multiple calls with the same Request produce equivalent Decisions."""
    req = make_request(
        "search docs for error logs",
        parameters={"query": "error logs"},
        constraints={"max_results": 5},
    )
    d1 = engine.decide(req)
    d2 = engine.decide(req)
    assert d1 == d2
    assert d1.primary_goal == d2.primary_goal
    assert d1.required_capabilities == d2.required_capabilities
    assert d1.execution_mode == d2.execution_mode
    assert d1.confidence == d2.confidence
    assert d1.reasoning == d2.reasoning
    assert d1.routing_hints == d2.routing_hints
