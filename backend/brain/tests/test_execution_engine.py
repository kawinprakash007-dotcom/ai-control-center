import pytest
from typing import List, Dict, Any, Optional

from core.interfaces.execution_engine_interface import ExecutionEngineInterface
from core.models.plan import Plan
from core.models.task import Task
from core.models.result import Result
from brain.execution import StandardExecutionEngine


class StubRouter:
    """
    Test stub for Router allowing controlled task results without executing real desktop tools.
    """

    def __init__(self, responses: Optional[Dict[int, Result]] = None, default_result: Optional[Result] = None):
        self.responses = responses or {}
        self.default_result = default_result or Result(success=True, message="Task completed.", output="default_output")
        self.routed_tasks: List[Task] = []
        self.raise_on_task_id: Optional[int] = None

    def route(self, task: Task) -> Result:
        self.routed_tasks.append(task)
        if self.raise_on_task_id is not None and task.id == self.raise_on_task_id:
            raise RuntimeError(f"Unexpected router error for task {task.id}")
        return self.responses.get(task.id, self.default_result)


def make_task(task_id: int = 1, tool: str = "chat", task_type: str = "chat", parameters: dict = None) -> Task:
    return Task(
        id=task_id,
        type=task_type,
        action=f"Execute {tool}",
        tool=tool,
        parameters=parameters or {},
        status="pending",
    )


def make_plan(steps: List[Task] = None, goal: str = "test_goal", confidence: float = 0.95) -> Plan:
    return Plan(
        goal=goal,
        steps=steps or [],
        status="pending",
        confidence=confidence,
    )


# ============================================================================
# 1. INTERFACE COMPLIANCE
# ============================================================================

def test_interface_compliance():
    """1. StandardExecutionEngine must implement ExecutionEngineInterface."""
    engine = StandardExecutionEngine(router=StubRouter())
    assert isinstance(engine, ExecutionEngineInterface)
    assert hasattr(engine, "execute")
    assert callable(engine.execute)


# ============================================================================
# 2. TYPE SAFETY
# ============================================================================

def test_type_safety_non_plan_input():
    """2. Passing non-Plan input raises TypeError."""
    engine = StandardExecutionEngine(router=StubRouter())
    with pytest.raises(TypeError) as excinfo:
        engine.execute("not a plan")  # type: ignore
    assert "expects a Plan instance" in str(excinfo.value)

    with pytest.raises(TypeError):
        engine.execute({"steps": []})  # type: ignore

    with pytest.raises(TypeError):
        engine.execute(None)  # type: ignore


# ============================================================================
# 3. EMPTY PLAN
# ============================================================================

def test_empty_plan_execution():
    """3. Empty Plan returns [] and sets plan.status = 'completed'."""
    router = StubRouter()
    engine = StandardExecutionEngine(router=router)
    plan = make_plan(steps=[])

    results = engine.execute(plan)

    assert results == []
    assert plan.status == "completed"
    assert len(router.routed_tasks) == 0


# ============================================================================
# 4. SINGLE TASK SUCCESS
# ============================================================================

def test_single_task_success():
    """4. Single task success updates task and plan status, populates task.result."""
    router = StubRouter(default_result=Result(success=True, message="Done", output="Hello World"))
    engine = StandardExecutionEngine(router=router)

    task = make_task(task_id=1, tool="chat", task_type="chat")
    plan = make_plan(steps=[task])

    results = engine.execute(plan)

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].output == "Hello World"
    assert task.status == "completed"
    assert task.result == "Hello World"
    assert plan.status == "completed"
    assert len(router.routed_tasks) == 1


# ============================================================================
# 5. MULTI-TASK SUCCESS & 13. EXECUTION ORDERING
# ============================================================================

def test_multi_task_success_ordering():
    """5 & 13. Multiple tasks execute sequentially in exact order."""
    t1 = make_task(task_id=1, tool="knowledge", task_type="knowledge")
    t2 = make_task(task_id=2, tool="notepad", task_type="tool")
    t3 = make_task(task_id=3, tool="chat", task_type="chat")
    plan = make_plan(steps=[t1, t2, t3])

    responses = {
        1: Result(success=True, message="Retrieved", output="doc_content"),
        2: Result(success=True, message="Opened", output="notepad_open"),
        3: Result(success=True, message="Responded", output="chat_response"),
    }
    router = StubRouter(responses=responses)
    engine = StandardExecutionEngine(router=router)

    results = engine.execute(plan)

    assert len(results) == 3
    assert [r.output for r in results] == ["doc_content", "notepad_open", "chat_response"]
    assert [t.status for t in plan.steps] == ["completed", "completed", "completed"]
    assert plan.status == "completed"
    assert [t.id for t in router.routed_tasks] == [1, 2, 3]


# ============================================================================
# 6. FAIL-FAST BEHAVIOR & UNEXECUTED TASKS SKIPPED
# ============================================================================

def test_fail_fast_behavior():
    """6. When task 2 fails, task 3 is skipped, exactly 2 results returned, plan fails."""
    t1 = make_task(task_id=1, tool="knowledge")
    t2 = make_task(task_id=2, tool="calculator")
    t3 = make_task(task_id=3, tool="chat")
    plan = make_plan(steps=[t1, t2, t3])

    responses = {
        1: Result(success=True, message="OK", output="data"),
        2: Result(success=False, message="Calculator execution failed", output=None),
        3: Result(success=True, message="OK", output="chat"),
    }
    router = StubRouter(responses=responses)
    engine = StandardExecutionEngine(router=router)

    results = engine.execute(plan)

    # Exactly 2 results collected
    assert len(results) == 2
    assert results[0].success is True
    assert results[1].success is False

    # Status checks
    assert t1.status == "completed"
    assert t2.status == "failed"
    assert t3.status == "skipped"
    assert plan.status == "failed"

    # Router was called only for tasks 1 and 2
    assert [t.id for t in router.routed_tasks] == [1, 2]


# ============================================================================
# 7. ROUTER RESULT PROPAGATION
# ============================================================================

def test_router_result_propagation():
    """7. The exact Result returned by the Router is preserved."""
    exact_result = Result(success=True, message="Custom message", output="Special Payload 123")
    router = StubRouter(default_result=exact_result)
    engine = StandardExecutionEngine(router=router)

    task = make_task(task_id=1)
    plan = make_plan(steps=[task])

    results = engine.execute(plan)
    assert len(results) == 1
    assert results[0] is exact_result
    assert results[0].message == "Custom message"
    assert results[0].output == "Special Payload 123"


# ============================================================================
# 8. KNOWLEDGE-SHAPED TASK
# ============================================================================

def test_knowledge_shaped_task():
    """8. Knowledge-shaped Task is passed unchanged to Router."""
    router = StubRouter()
    engine = StandardExecutionEngine(router=router)

    params = {"query": "explain linux virtual memory", "max_results": 5}
    task = make_task(task_id=1, tool="knowledge", task_type="knowledge", parameters=params)
    plan = make_plan(steps=[task])

    engine.execute(plan)

    assert len(router.routed_tasks) == 1
    routed = router.routed_tasks[0]
    assert routed.type == "knowledge"
    assert routed.tool == "knowledge"
    assert routed.parameters["query"] == "explain linux virtual memory"
    assert routed.parameters["max_results"] == 5


# ============================================================================
# 9. TOOL-SHAPED TASK
# ============================================================================

def test_tool_shaped_task():
    """9. Task.tool is passed unchanged to Router."""
    router = StubRouter()
    engine = StandardExecutionEngine(router=router)

    task = make_task(task_id=1, tool="OpenCV", task_type="tool")
    plan = make_plan(steps=[task])

    engine.execute(plan)

    assert len(router.routed_tasks) == 1
    assert router.routed_tasks[0].tool == "OpenCV"


# ============================================================================
# 10. PARAMETER PRESERVATION
# ============================================================================

def test_parameter_preservation():
    """10. Execution Engine does not mutate task.parameters."""
    router = StubRouter()
    engine = StandardExecutionEngine(router=router)

    original_params = {"key1": "val1", "nested": {"a": 1}}
    task = make_task(task_id=1, parameters=dict(original_params))
    plan = make_plan(steps=[task])

    engine.execute(plan)

    assert task.parameters == original_params


# ============================================================================
# 11 & 12. PLAN GOAL & CONFIDENCE PRESERVATION
# ============================================================================

def test_plan_goal_and_confidence_preservation():
    """11 & 12. Plan goal and confidence are strictly preserved."""
    router = StubRouter()
    engine = StandardExecutionEngine(router=router)

    plan = make_plan(
        steps=[make_task(task_id=1)],
        goal="retrieve_kernel_docs",
        confidence=0.88,
    )

    engine.execute(plan)

    assert plan.goal == "retrieve_kernel_docs"
    assert plan.confidence == 0.88


# ============================================================================
# 14. ROUTER DEPENDENCY INJECTION
# ============================================================================

def test_router_dependency_injection():
    """14. Injected Router instance is utilized for execution."""
    custom_router = StubRouter(default_result=Result(success=True, message="Injected router ran"))
    engine = StandardExecutionEngine(router=custom_router)
    assert engine.router is custom_router

    plan = make_plan(steps=[make_task(1)])
    results = engine.execute(plan)

    assert results[0].message == "Injected router ran"
    assert len(custom_router.routed_tasks) == 1


# ============================================================================
# 15. NO DIRECT TOOL EXECUTION
# ============================================================================

def test_no_direct_tool_execution(monkeypatch):
    """15. Execution Engine delegates to router.route and does not execute tools directly."""
    # Ensure StandardExecutionEngine module does not import or call tools directly
    import brain.execution.standard_execution_engine as see_module
    assert not hasattr(see_module, "open_calculator")
    assert not hasattr(see_module, "open_notepad")
    assert not hasattr(see_module, "TOOL_REGISTRY")


# ============================================================================
# 16. UNEXPECTED ROUTER EXCEPTION HANDLING
# ============================================================================

def test_unexpected_router_exception_handled():
    """16. Unexpected exception in router is caught safely and converted to failure Result."""
    router = StubRouter()
    router.raise_on_task_id = 2

    t1 = make_task(task_id=1)
    t2 = make_task(task_id=2)
    t3 = make_task(task_id=3)
    plan = make_plan(steps=[t1, t2, t3])

    engine = StandardExecutionEngine(router=router)
    results = engine.execute(plan)

    assert len(results) == 2
    assert results[0].success is True
    assert results[1].success is False
    assert "Router execution exception" in results[1].message

    assert t1.status == "completed"
    assert t2.status == "failed"
    assert t3.status == "skipped"
    assert plan.status == "failed"


# ============================================================================
# 17. TASK RESULT FIELD POPULATED
# ============================================================================

def test_task_result_field_populated_message_fallback():
    """17. If result.output is empty/None, result.message is populated onto task.result."""
    router = StubRouter(default_result=Result(success=True, message="Action Done Without Output", output=None))
    engine = StandardExecutionEngine(router=router)

    task = make_task(task_id=1)
    plan = make_plan(steps=[task])

    engine.execute(plan)

    assert task.status == "completed"
    assert task.result == "Action Done Without Output"
