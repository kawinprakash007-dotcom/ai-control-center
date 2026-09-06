import pytest
import sys
import dataclasses

from core.interfaces.decision_planner_interface import DecisionPlannerInterface
from core.models.decision import Decision, CapabilityType, ExecutionMode
from core.models.plan import Plan
from core.models.task import Task
from brain.planning import StandardPlanner


@pytest.fixture
def planner():
    return StandardPlanner()


def make_decision(
    primary_goal: str = "answer_chat",
    required_capabilities: list = None,
    execution_mode: ExecutionMode = ExecutionMode.DIRECT,
    confidence: float = 1.0,
    reasoning: str = "test decision",
    routing_hints: dict = None,
    request_id: str = "req-101",
) -> Decision:
    return Decision(
        request_id=request_id,
        primary_goal=primary_goal,
        required_capabilities=required_capabilities or [CapabilityType.CHAT],
        execution_mode=execution_mode,
        confidence=confidence,
        reasoning=reasoning,
        routing_hints=routing_hints or {},
    )


# ============================================================================
# 1. INTERFACE COMPLIANCE
# ============================================================================

def test_interface_compliance(planner):
    """1. StandardPlanner must implement DecisionPlannerInterface."""
    assert isinstance(planner, DecisionPlannerInterface)
    assert hasattr(planner, "plan")
    assert callable(planner.plan)


# ============================================================================
# 2. TYPE SAFETY FOR INPUT
# ============================================================================

def test_type_error_for_non_decision_input(planner):
    """2 & 18. Passing non-Decision input raises TypeError."""
    with pytest.raises(TypeError) as excinfo:
        planner.plan("not a decision")  # type: ignore
    assert "expects a Decision instance" in str(excinfo.value)

    with pytest.raises(TypeError):
        planner.plan({"primary_goal": "chat"})  # type: ignore

    with pytest.raises(TypeError):
        planner.plan(None)  # type: ignore


# ============================================================================
# 3. DECISION -> PLAN CORRELATION
# ============================================================================

def test_decision_to_plan_correlation(planner):
    """3, 14, 15. Plan goal and confidence strictly correlate with Decision."""
    dec = make_decision(
        primary_goal="retrieve_knowledge",
        confidence=0.92,
        execution_mode=ExecutionMode.SINGLE_STEP,
        required_capabilities=[CapabilityType.KNOWLEDGE],
    )
    plan = planner.plan(dec)
    assert isinstance(plan, Plan)
    assert plan.goal == "retrieve_knowledge"
    assert plan.confidence == 0.92
    assert plan.status == "pending"


# ============================================================================
# 4. DIRECT CHAT PLANNING
# ============================================================================

def test_direct_chat_planning(planner):
    """4. Direct chat produces a single chat task."""
    dec = make_decision(
        primary_goal="answer_chat",
        required_capabilities=[CapabilityType.CHAT],
        execution_mode=ExecutionMode.DIRECT,
    )
    plan = planner.plan(dec)
    assert len(plan.steps) == 1
    task = plan.steps[0]
    assert isinstance(task, Task)
    assert task.id == 1
    assert task.type == "chat"
    assert task.action == "Respond to User"
    assert task.tool == "chat"
    assert task.status == "pending"


# ============================================================================
# 5. EMPTY-INPUT PLANNING
# ============================================================================

def test_empty_input_planning(planner):
    """5. Empty-input decision produces zero steps (non-executable control-flow)."""
    dec = make_decision(
        primary_goal="prompt_user_input",
        required_capabilities=[CapabilityType.CHAT],
        execution_mode=ExecutionMode.DIRECT,
        confidence=1.0,
        routing_hints={"is_empty": True},
    )
    plan = planner.plan(dec)
    assert plan.steps == []
    assert len(plan.steps) == 0
    assert plan.goal == "prompt_user_input"
    assert plan.confidence == 1.0
    assert plan.status == "pending"


# ============================================================================
# 6. CLARIFICATION PLANNING
# ============================================================================

def test_clarification_planning(planner):
    """6. Clarification decision produces zero steps (non-executable control-flow)."""
    dec = make_decision(
        primary_goal="clarify_request",
        required_capabilities=[CapabilityType.CHAT],
        execution_mode=ExecutionMode.DIRECT,
        confidence=0.95,
        routing_hints={"ambiguity_reason": "Input contains only punctuation"},
    )
    plan = planner.plan(dec)
    assert plan.steps == []
    assert len(plan.steps) == 0
    assert plan.goal == "clarify_request"
    assert plan.confidence == 0.95
    assert plan.status == "pending"


def test_control_flow_decisions_do_not_create_chat_tasks(planner):
    """Verify neither prompt_user_input nor clarify_request creates any chat task."""
    for goal in ("prompt_user_input", "clarify_request"):
        dec = make_decision(
            primary_goal=goal,
            required_capabilities=[CapabilityType.CHAT],
            execution_mode=ExecutionMode.DIRECT,
        )
        plan = planner.plan(dec)
        assert len(plan.steps) == 0
        assert not any(t.type == "chat" or t.tool == "chat" for t in plan.steps)


# ============================================================================
# 7. KNOWLEDGE PLANNING
# ============================================================================

def test_knowledge_planning(planner):
    """7. Knowledge decision produces a retrieve_knowledge task with query parameters."""
    dec = make_decision(
        primary_goal="retrieve_knowledge",
        required_capabilities=[CapabilityType.KNOWLEDGE],
        execution_mode=ExecutionMode.SINGLE_STEP,
        routing_hints={"query": "linux kernel vfs", "max_results": 5},
    )
    plan = planner.plan(dec)
    assert len(plan.steps) == 1
    task = plan.steps[0]
    assert task.id == 1
    assert task.type == "knowledge"
    assert task.action == "Retrieve Knowledge"
    assert task.tool == "knowledge"
    assert task.parameters["query"] == "linux kernel vfs"
    assert task.parameters["max_results"] == 5


# ============================================================================
# 8. TOOL PLANNING
# ============================================================================

def test_tool_planning(planner):
    """8. Tool decision produces execute_tool task preserving tool_hint casing exactly."""
    dec = make_decision(
        primary_goal="execute_tool",
        required_capabilities=[CapabilityType.TOOL],
        execution_mode=ExecutionMode.SINGLE_STEP,
        routing_hints={"tool_hint": "OpenCV"},
    )
    plan = planner.plan(dec)
    assert len(plan.steps) == 1
    task = plan.steps[0]
    assert task.id == 1
    assert task.type == "tool"
    assert task.tool == "OpenCV"
    assert task.action == "Execute Tool"


# ============================================================================
# 9. MEMORY PLANNING
# ============================================================================

def test_memory_planning(planner):
    """9. Memory decision produces manage_memory task."""
    dec = make_decision(
        primary_goal="manage_memory",
        required_capabilities=[CapabilityType.MEMORY],
        execution_mode=ExecutionMode.SINGLE_STEP,
    )
    plan = planner.plan(dec)
    assert len(plan.steps) == 1
    task = plan.steps[0]
    assert task.id == 1
    assert task.type == "memory"
    assert task.tool == "memory"
    assert task.action == "Manage Memory"


# ============================================================================
# 10. MULTI-STEP PLANNING & 11. TASK ID SEQUENCING
# ============================================================================

def test_multi_step_planning_and_sequencing(planner):
    """10 & 11. Multi-step decision produces sequential Task steps (1, 2, 3)."""
    dec = make_decision(
        primary_goal="execute_workflow",
        required_capabilities=[
            CapabilityType.KNOWLEDGE,
            CapabilityType.TOOL,
            CapabilityType.CHAT,
        ],
        execution_mode=ExecutionMode.MULTI_STEP,
        routing_hints={"query": "find bug reports", "tool_hint": "notepad"},
    )
    plan = planner.plan(dec)
    assert len(plan.steps) == 3

    # Step 1: Knowledge
    t1 = plan.steps[0]
    assert t1.id == 1
    assert t1.type == "knowledge"
    assert t1.tool == "knowledge"
    assert t1.parameters["query"] == "find bug reports"

    # Step 2: Tool
    t2 = plan.steps[1]
    assert t2.id == 2
    assert t2.type == "tool"
    assert t2.tool == "notepad"

    # Step 3: Chat
    t3 = plan.steps[2]
    assert t3.id == 3
    assert t3.type == "chat"
    assert t3.tool == "chat"


# ============================================================================
# 12. ROUTING HINT PROPAGATION & 13. PARAMETER INDEPENDENCE
# ============================================================================

def test_routing_hint_propagation_and_independence(planner):
    """12 & 13. Routing hints propagate into independent task dictionaries."""
    hints = {
        "query": "system architecture",
        "path": "C:\\logs\\app.log",
        "max_results": 10,
        "timeout_seconds": 30.0,
        "privacy_level": "local_only",
        "confirmation_required": True,
    }
    dec = make_decision(
        primary_goal="execute_workflow",
        required_capabilities=[CapabilityType.KNOWLEDGE, CapabilityType.TOOL],
        execution_mode=ExecutionMode.MULTI_STEP,
        routing_hints=hints,
    )
    plan = planner.plan(dec)
    assert len(plan.steps) == 2

    # Verify propagation
    for task in plan.steps:
        for k, v in hints.items():
            assert task.parameters[k] == v

    # Verify dictionary independence (modifying one task doesn't mutate another or decision)
    plan.steps[0].parameters["custom_mutation"] = "mutated"
    assert "custom_mutation" not in plan.steps[1].parameters
    assert "custom_mutation" not in dec.routing_hints


# ============================================================================
# 16. DECISION IMMUTABILITY
# ============================================================================

def test_decision_immutability(planner):
    """16. Decision remains strictly immutable and unmutated after planning."""
    hints = {"query": "immutable test"}
    dec = make_decision(
        primary_goal="retrieve_knowledge",
        execution_mode=ExecutionMode.SINGLE_STEP,
        required_capabilities=[CapabilityType.KNOWLEDGE],
        routing_hints=hints,
    )
    _ = planner.plan(dec)

    with pytest.raises(dataclasses.FrozenInstanceError):
        dec.primary_goal = "modified"  # type: ignore

    assert dec.routing_hints == {"query": "immutable test"}


# ============================================================================
# 17. DETERMINISM
# ============================================================================

def test_planner_determinism(planner):
    """17. Planning with the same Decision yields identical Plan output."""
    dec = make_decision(
        primary_goal="execute_workflow",
        required_capabilities=[CapabilityType.KNOWLEDGE, CapabilityType.TOOL],
        execution_mode=ExecutionMode.MULTI_STEP,
        routing_hints={"query": "test query", "tool_hint": "calc"},
    )
    p1 = planner.plan(dec)
    p2 = planner.plan(dec)

    assert p1.goal == p2.goal
    assert p1.confidence == p2.confidence
    assert p1.status == p2.status
    assert len(p1.steps) == len(p2.steps)
    for s1, s2 in zip(p1.steps, p2.steps):
        assert s1.id == s2.id
        assert s1.type == s2.type
        assert s1.action == s2.action
        assert s1.tool == s2.tool
        assert s1.parameters == s2.parameters


# ============================================================================
# 19. UNSUPPORTED CAPABILITY HANDLING
# ============================================================================

def test_unsupported_capability_handling(planner):
    """19. Capabilities without execution mappings (VISION, DEVICE) fail with ValueError."""
    for unsupported_cap in (CapabilityType.VISION, CapabilityType.DEVICE):
        dec = make_decision(
            primary_goal="test_unsupported",
            required_capabilities=[unsupported_cap],
            execution_mode=ExecutionMode.MULTI_STEP,
        )
        with pytest.raises(ValueError) as excinfo:
            planner.plan(dec)
        assert "does not yet have a supported execution mapping" in str(excinfo.value)


def test_web_capability_planning(planner):
    """19b. Web capability produces valid single-step web Task."""
    dec = make_decision(
        primary_goal="web_search",
        required_capabilities=[CapabilityType.WEB],
        execution_mode=ExecutionMode.SINGLE_STEP,
        routing_hints={"action": "search", "query": "Python 3.14 release"},
    )
    plan = planner.plan(dec)
    assert len(plan.steps) == 1
    task = plan.steps[0]
    assert task.type == "web"
    assert task.tool == "web"
    assert task.action == "Web Search"
    assert task.parameters["action"] == "search"
    assert task.parameters["query"] == "Python 3.14 release"


# ============================================================================
# 20. NO LEGACY GOAL / PLANNER DEPENDENCY
# ============================================================================

def test_no_legacy_goal_dependency():
    """20. StandardPlanner does not import or depend on legacy planner.py or Goal."""
    # Ensure brain.planning does not import brain.planner
    import brain.planning.standard_planner as sp_module

    source_text = open(sp_module.__file__, "r").read()
    assert "brain.planner" not in source_text
    assert "core.models.goal" not in source_text
    assert "GoalManager" not in source_text
    assert "Goal" not in source_text
