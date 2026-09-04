import pytest
from datetime import datetime
import dataclasses

from core.models.request import Request
from core.models.decision import (
    CapabilityType,
    ExecutionMode,
    CapabilityRequirement,
    Decision,
)
from core.interfaces.request_understanding_interface import RequestUnderstandingInterface
from core.interfaces.decision_engine_interface import DecisionEngineInterface


# ============================================================================
# REQUEST MODEL TESTS
# ============================================================================

def test_request_instantiation():
    """1. Request can be instantiated."""
    now = datetime.now()
    req = Request(
        id="req-001",
        original_text="What is a Linux process?",
        normalized_text="what is a linux process",
        session_id="session-123",
        timestamp=now
    )
    assert req is not None
    assert isinstance(req, Request)


def test_request_fields_stored_correctly():
    """2. All fields are stored correctly."""
    now = datetime.now()
    params = {"timeout": 30, "domain": "linux"}
    constraints = {"max_tokens": 512}
    req = Request(
        id="req-002",
        original_text="Raw Text Here",
        normalized_text="raw text here",
        session_id="sess-xyz",
        timestamp=now,
        source="api",
        parameters=params,
        constraints=constraints
    )
    assert req.id == "req-002"
    assert req.original_text == "Raw Text Here"
    assert req.normalized_text == "raw text here"
    assert req.session_id == "sess-xyz"
    assert req.timestamp == now
    assert req.source == "api"
    assert req.parameters == params
    assert req.constraints == constraints


def test_request_default_source():
    """3. Default source is 'chat'."""
    req = Request(
        id="req-003",
        original_text="Hello",
        normalized_text="hello",
        session_id="sess-1",
        timestamp=datetime.now()
    )
    assert req.source == "chat"


def test_request_default_parameters_empty_dict():
    """4. parameters default to an empty dictionary."""
    req = Request(
        id="req-004",
        original_text="Hello",
        normalized_text="hello",
        session_id="sess-1",
        timestamp=datetime.now()
    )
    assert req.parameters == {}
    assert isinstance(req.parameters, dict)


def test_request_default_constraints_empty_dict():
    """5. constraints default to an empty dictionary."""
    req = Request(
        id="req-005",
        original_text="Hello",
        normalized_text="hello",
        session_id="sess-1",
        timestamp=datetime.now()
    )
    assert req.constraints == {}
    assert isinstance(req.constraints, dict)


def test_request_independent_default_dictionaries():
    """6. Two Request objects do not share the same default dictionaries."""
    now = datetime.now()
    req1 = Request(
        id="req-1",
        original_text="One",
        normalized_text="one",
        session_id="s1",
        timestamp=now
    )
    req2 = Request(
        id="req-2",
        original_text="Two",
        normalized_text="two",
        session_id="s2",
        timestamp=now
    )
    assert req1.parameters is not req2.parameters
    assert req1.constraints is not req2.constraints


def test_request_frozen_immutability():
    """7. frozen=True prevents field reassignment."""
    req = Request(
        id="req-007",
        original_text="Immutable text",
        normalized_text="immutable text",
        session_id="sess-1",
        timestamp=datetime.now()
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.original_text = "Mutated text"  # type: ignore

    with pytest.raises(dataclasses.FrozenInstanceError):
        req.id = "new-id"  # type: ignore


def test_request_original_text_separate_from_normalized():
    """8. original_text remains unchanged when normalized_text differs."""
    original = "   WHAT IS A LINUX PROCESS?!!   \n"
    normalized = "what is a linux process"
    req = Request(
        id="req-008",
        original_text=original,
        normalized_text=normalized,
        session_id="s1",
        timestamp=datetime.now()
    )
    assert req.original_text == original
    assert req.normalized_text == normalized
    assert req.original_text != req.normalized_text


def test_request_asdict_serialization():
    """9. serialization using dataclasses.asdict() works."""
    now = datetime.now()
    req = Request(
        id="req-009",
        original_text="Sample text",
        normalized_text="sample text",
        session_id="s9",
        timestamp=now,
        source="voice",
        parameters={"audio_format": "wav"},
        constraints={"timeout_sec": 5}
    )
    d = dataclasses.asdict(req)
    assert isinstance(d, dict)
    assert d["id"] == "req-009"
    assert d["original_text"] == "Sample text"
    assert d["normalized_text"] == "sample text"
    assert d["session_id"] == "s9"
    assert d["timestamp"] == now
    assert d["source"] == "voice"
    assert d["parameters"] == {"audio_format": "wav"}
    assert d["constraints"] == {"timeout_sec": 5}


# ============================================================================
# DECISION MODEL TESTS
# ============================================================================

def test_capability_type_contains_all_seven_values():
    """10. CapabilityType contains all seven capability values."""
    expected = {
        "chat": "chat",
        "knowledge": "knowledge",
        "web": "web",
        "memory": "memory",
        "tool": "tool",
        "vision": "vision",
        "device": "device"
    }
    actual = {member.name.lower(): member.value for member in CapabilityType}
    assert actual == expected
    assert len(CapabilityType) == 7


def test_execution_mode_values():
    """11. ExecutionMode contains DIRECT, SINGLE_STEP, MULTI_STEP."""
    expected = {
        "DIRECT": "direct",
        "SINGLE_STEP": "single_step",
        "MULTI_STEP": "multi_step"
    }
    actual = {member.name: member.value for member in ExecutionMode}
    assert actual == expected
    assert len(ExecutionMode) == 3


def test_capability_requirement_instantiation():
    """12. CapabilityRequirement can be instantiated."""
    req = CapabilityRequirement(capability=CapabilityType.KNOWLEDGE)
    assert req is not None
    assert req.capability == CapabilityType.KNOWLEDGE


def test_capability_requirement_default_priority():
    """13. CapabilityRequirement default priority is 1."""
    req = CapabilityRequirement(capability=CapabilityType.CHAT)
    assert req.priority == 1


def test_capability_requirement_default_mandatory():
    """14. CapabilityRequirement default mandatory is True."""
    req = CapabilityRequirement(capability=CapabilityType.TOOL)
    assert req.mandatory is True


def test_capability_requirement_independent_parameters():
    """15. CapabilityRequirement parameters use independent dictionaries."""
    r1 = CapabilityRequirement(capability=CapabilityType.WEB)
    r2 = CapabilityRequirement(capability=CapabilityType.MEMORY)
    assert r1.parameters == {}
    assert r2.parameters == {}
    assert r1.parameters is not r2.parameters


def test_decision_instantiation():
    """16. Decision can be instantiated."""
    dec = Decision(
        request_id="req-100",
        primary_goal="answer_factual_question",
        required_capabilities=[CapabilityType.KNOWLEDGE],
        execution_mode=ExecutionMode.SINGLE_STEP,
        confidence=0.98,
        reasoning="User is requesting technical documentation."
    )
    assert dec is not None
    assert isinstance(dec, Decision)


def test_decision_fields_stored_correctly():
    """17. Decision fields are stored correctly."""
    hints = {"collection": "linux", "top_k": 3}
    dec = Decision(
        request_id="req-101",
        primary_goal="system_automation",
        required_capabilities=[CapabilityType.TOOL, CapabilityType.VISION],
        execution_mode=ExecutionMode.MULTI_STEP,
        confidence=0.88,
        reasoning="Requires reading screen and executing click.",
        routing_hints=hints
    )
    assert dec.request_id == "req-101"
    assert dec.primary_goal == "system_automation"
    assert dec.required_capabilities == [CapabilityType.TOOL, CapabilityType.VISION]
    assert dec.execution_mode == ExecutionMode.MULTI_STEP
    assert dec.confidence == 0.88
    assert dec.reasoning == "Requires reading screen and executing click."
    assert dec.routing_hints == hints


def test_decision_default_routing_hints_independent():
    """18. Decision default routing_hints is an independent dictionary."""
    d1 = Decision(
        request_id="req-1",
        primary_goal="goal1",
        required_capabilities=[CapabilityType.CHAT],
        execution_mode=ExecutionMode.DIRECT,
        confidence=1.0,
        reasoning="direct chat"
    )
    d2 = Decision(
        request_id="req-2",
        primary_goal="goal2",
        required_capabilities=[CapabilityType.TOOL],
        execution_mode=ExecutionMode.SINGLE_STEP,
        confidence=0.95,
        reasoning="run tool"
    )
    assert d1.routing_hints == {}
    assert d2.routing_hints == {}
    assert d1.routing_hints is not d2.routing_hints


def test_decision_frozen_immutability():
    """19. frozen=True prevents field reassignment."""
    dec = Decision(
        request_id="req-103",
        primary_goal="immutable_goal",
        required_capabilities=[CapabilityType.KNOWLEDGE],
        execution_mode=ExecutionMode.SINGLE_STEP,
        confidence=0.9,
        reasoning="cannot change"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        dec.primary_goal = "mutated_goal"  # type: ignore

    with pytest.raises(dataclasses.FrozenInstanceError):
        dec.confidence = 0.5  # type: ignore


def test_decision_asdict_serialization():
    """20. dataclasses.asdict() works for Decision."""
    dec = Decision(
        request_id="req-104",
        primary_goal="retrieve_kernel_docs",
        required_capabilities=[CapabilityType.KNOWLEDGE],
        execution_mode=ExecutionMode.SINGLE_STEP,
        confidence=0.92,
        reasoning="technical domain query",
        routing_hints={"collection": "linux"}
    )
    d = dataclasses.asdict(dec)
    assert isinstance(d, dict)
    assert d["request_id"] == "req-104"
    assert d["primary_goal"] == "retrieve_kernel_docs"
    assert d["required_capabilities"] == [CapabilityType.KNOWLEDGE]
    assert d["execution_mode"] == ExecutionMode.SINGLE_STEP
    assert d["confidence"] == 0.92
    assert d["reasoning"] == "technical domain query"
    assert d["routing_hints"] == {"collection": "linux"}


# ============================================================================
# INTERFACE TESTS
# ============================================================================

def test_request_understanding_interface_cannot_be_instantiated():
    """21. RequestUnderstandingInterface cannot be instantiated because it is abstract."""
    with pytest.raises(TypeError) as excinfo:
        RequestUnderstandingInterface()  # type: ignore
    assert "Can't instantiate abstract class" in str(excinfo.value)


def test_decision_engine_interface_cannot_be_instantiated():
    """22. DecisionEngineInterface cannot be instantiated because it is abstract."""
    with pytest.raises(TypeError) as excinfo:
        DecisionEngineInterface()  # type: ignore
    assert "Can't instantiate abstract class" in str(excinfo.value)
