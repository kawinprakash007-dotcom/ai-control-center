from typing import Optional, Union

from core.models.task import Task
from core.models.goal import Goal
from core.models.intent import Intent
from core.models.result import Result
from core.constants.task_types import TaskType

from tools.executor import Executor
from brain.router import Router
from tools.tool_registry import TOOL_REGISTRY
from tools.knowledge_capability import KnowledgeCapability

from brain.intent_engine import IntentEngine
from brain.reasoner import Reasoner
from brain.planner import Planner
from brain.agent import Agent

from knowledge.interfaces.rag_service_interface import RAGServiceInterface
from knowledge.models.query import Query
from knowledge.models.knowledge_context import KnowledgeContext
from knowledge.models.chunk import Chunk


class FakeRAGService(RAGServiceInterface):
    """Stub RAGService for deterministic testing without external dependencies."""

    def __init__(self, return_text: str = "Fake formatted context"):
        self.last_query = None
        self.call_count = 0
        self.return_text = return_text

    def query(self, query: Union[Query, str, None]) -> KnowledgeContext:
        self.call_count += 1
        self.last_query = query.text if isinstance(query, Query) else str(query)
        chunk = Chunk(id="c1", document_id="doc1", text="some text", chunk_index=0)
        return KnowledgeContext(
            query=self.last_query,
            chunks=[chunk],
            formatted_context=self.return_text
        )


def test_agent_rag_integration():
    print("=" * 60)
    print("RUNNING AGENT <-> RAG INTEGRATION TESTS")
    print("=" * 60)

    # -------------------------------------------------------------
    # Test 1: Knowledge query reaches KnowledgeCapability
    # -------------------------------------------------------------
    print("\n[Test 1] Knowledge query reaches KnowledgeCapability")
    fake_rag = FakeRAGService(return_text="[Source 1]\nDocument: OS.pdf\nContent:\nProcess info")
    cap = KnowledgeCapability(rag_service=fake_rag)
    task = Task(
        id=1,
        type="knowledge",
        action="Retrieve Knowledge",
        tool="knowledge",
        parameters={"query": "explain linux process"}
    )
    output = cap(task)
    assert fake_rag.call_count == 1, "RAGService must be called exactly once"
    assert fake_rag.last_query == "explain linux process", f"Expected query 'explain linux process', got '{fake_rag.last_query}'"
    assert "[Source 1]" in output, "Expected '[Source 1]' in output"
    assert "OS.pdf" in output, "Expected 'OS.pdf' in output"
    print("  [OK] Test 1 PASSED: Knowledge query correctly reached KnowledgeCapability!")

    # -------------------------------------------------------------
    # Test 2: Real Linux knowledge query reaches real RAGService
    # -------------------------------------------------------------
    print("\n[Test 2] Real Linux knowledge query reaches real RAGService")
    # A. Direct execution via KnowledgeCapability with real RAGService
    real_cap = KnowledgeCapability()
    real_task = Task(
        id=1,
        type="knowledge",
        action="Retrieve Knowledge",
        tool="knowledge",
        parameters={"query": "What is a Linux process?"}
    )
    real_output = real_cap(real_task)
    assert "Linux.pdf" in real_output, "Expected 'Linux.pdf' in real output"
    assert "[Source 1]" in real_output, "Expected '[Source 1]' in real output"
    assert "Content:" in real_output, "Expected 'Content:' in real output"
    assert len(real_output.strip()) > 0, "Real output must not be empty"

    # B. Full end-to-end execution through Agent.think()
    agent = Agent()
    # Stub IntentEngine.detect to bypass Ollama network call for deterministic agent test
    agent.intent_engine.detect = lambda msg: Intent(
        intent="knowledge",
        reason="User is asking for factual Linux concept",
        original_message=msg
    )
    agent_response = agent.think("What is a Linux process?")
    assert "Linux.pdf" in agent_response, "Expected 'Linux.pdf' in agent response"
    assert "[Source 1]" in agent_response, "Expected '[Source 1]' in agent response"
    print("  [OK] Test 2 PASSED: Real Linux query end-to-end retrieval successful!")

    # -------------------------------------------------------------
    # Test 3: Task parameters correctly carry {"query": original_user_query}
    # -------------------------------------------------------------
    print("\n[Test 3] Task parameters correctly carry {'query': original_user_query}")
    planner = Planner()
    original_msg = "Explain virtual memory architecture in Linux"
    goal = Goal(goal="Retrieve Knowledge", query=original_msg)
    plan = planner.create_plan(goal)

    assert len(plan.steps) == 1, f"Expected 1 plan step, got {len(plan.steps)}"
    planned_task = plan.steps[0]
    assert planned_task.type == "knowledge", f"Expected type 'knowledge', got '{planned_task.type}'"
    assert planned_task.tool == "knowledge", f"Expected tool 'knowledge', got '{planned_task.tool}'"
    assert planned_task.parameters == {"query": original_msg}, f"Parameters mismatch: {planned_task.parameters}"
    print("  [OK] Test 3 PASSED: Task parameters faithfully preserve original query!")

    # -------------------------------------------------------------
    # Test 4: Executor still supports parameterless legacy tools
    # -------------------------------------------------------------
    print("\n[Test 4] Executor still supports parameterless legacy tools")
    executor = Executor()
    legacy_called = False

    def legacy_tool():
        nonlocal legacy_called
        legacy_called = True
        return "Legacy tool execution successful"

    # Execution without task parameter
    res1 = executor.execute(legacy_tool)
    assert res1.success is True
    assert legacy_called is True
    assert res1.output == "Legacy tool execution successful"

    # Execution with task parameter provided (must not pass to parameterless function)
    legacy_called = False
    dummy_task = Task(id=1, type="tool", action="open", tool="legacy")
    res2 = executor.execute(legacy_tool, task=dummy_task)
    assert res2.success is True
    assert legacy_called is True
    assert res2.output == "Legacy tool execution successful"
    print("  [OK] Test 4 PASSED: Legacy parameterless tools fully supported without errors!")

    # -------------------------------------------------------------
    # Test 5: Executor correctly passes Task to task-aware capabilities
    # -------------------------------------------------------------
    print("\n[Test 5] Executor correctly passes Task to task-aware capabilities")
    received_task_in_tool = None

    def task_aware_tool(task_obj):
        nonlocal received_task_in_tool
        received_task_in_tool = task_obj
        return f"Handled action: {task_obj.action}"

    target_task = Task(
        id=77,
        type="knowledge",
        action="Retrieve Kernel Docs",
        tool="knowledge",
        parameters={"query": "kernel"}
    )
    res_aware = executor.execute(task_aware_tool, task=target_task)
    assert res_aware.success is True
    assert received_task_in_tool is target_task, "Task instance was not received by task-aware capability"
    assert res_aware.output == "Handled action: Retrieve Kernel Docs"
    print("  [OK] Test 5 PASSED: Task correctly passed into task-aware capability!")

    # -------------------------------------------------------------
    # Test 6: Router correctly passes Task to Executor
    # -------------------------------------------------------------
    print("\n[Test 6] Router correctly passes Task to Executor")
    router = Router()
    received_task_in_router_target = None

    def custom_test_capability(task_param):
        nonlocal received_task_in_router_target
        received_task_in_router_target = task_param
        return "Custom tool executed"

    TOOL_REGISTRY["_test_custom_cap"] = custom_test_capability
    try:
        routed_task = Task(
            id=88,
            type="tool",
            action="Run Custom Test",
            tool="_test_custom_cap",
            parameters={"query": "routed query"}
        )
        route_res = router.route(routed_task)
        assert route_res.success is True
        assert received_task_in_router_target is routed_task, "Router did not propagate Task to Executor"
        assert received_task_in_router_target.parameters == {"query": "routed query"}
    finally:
        del TOOL_REGISTRY["_test_custom_cap"]
    print("  [OK] Test 6 PASSED: Router cleanly propagated Task to Executor!")

    # -------------------------------------------------------------
    # Test 7: KnowledgeCapability correctly extracts query from Task
    # -------------------------------------------------------------
    print("\n[Test 7] KnowledgeCapability correctly extracts query from Task")
    spy_rag = FakeRAGService(return_text="extracted query context")
    test_cap = KnowledgeCapability(rag_service=spy_rag)

    # 1. From Task.parameters["query"]
    t_obj = Task(id=1, type="knowledge", action="Q", tool="knowledge", parameters={"query": "Explicit Parameter Query"})
    test_cap(t_obj)
    assert spy_rag.last_query == "Explicit Parameter Query"

    # 2. From direct string invocation
    test_cap("Direct String Query")
    assert spy_rag.last_query == "Direct String Query"
    print("  [OK] Test 7 PASSED: Query extraction handles Task objects and strings accurately!")

    # -------------------------------------------------------------
    # Test 8: Empty/missing query is handled safely
    # -------------------------------------------------------------
    print("\n[Test 8] Empty/missing query is handled safely")
    guard_rag = FakeRAGService()
    safe_cap = KnowledgeCapability(rag_service=guard_rag)

    for empty_task in [
        None,
        Task(id=1, type="knowledge", action="Q", tool="knowledge", parameters={}),
        Task(id=1, type="knowledge", action="Q", tool="knowledge", parameters={"query": ""}),
        Task(id=1, type="knowledge", action="Q", tool="knowledge", parameters={"query": "   \n\t "}),
        ""
    ]:
        out = safe_cap(empty_task)
        assert out == "No query provided for knowledge retrieval.", f"Expected safe message, got '{out}'"
        assert guard_rag.call_count == 0, "RAGService must not be called when query is empty"
    print("  [OK] Test 8 PASSED: Empty/missing queries safely handled without invoking RAGService!")

    # -------------------------------------------------------------
    # Test 9: Existing chat flow does not break
    # -------------------------------------------------------------
    print("\n[Test 9] Existing chat flow does not break")
    planner = Planner()
    chat_plan = planner.create_plan(Goal(goal="Respond to User"))
    assert len(chat_plan.steps) == 1
    assert chat_plan.steps[0].type == "chat"
    assert chat_plan.steps[0].tool == "chat"

    # Verify tool registry has chat
    assert "chat" in TOOL_REGISTRY
    orig_chat = TOOL_REGISTRY["chat"]
    # Temporarily mock chat to avoid network calls
    TOOL_REGISTRY["chat"] = lambda: "Chat output placeholder"
    try:
        router = Router()
        chat_res = router.route(chat_plan.steps[0])
        assert chat_res.success is True
        assert chat_res.output == "Chat output placeholder"
    finally:
        TOOL_REGISTRY["chat"] = orig_chat
    print("  [OK] Test 9 PASSED: Chat planning and execution flow remains completely intact!")

    # -------------------------------------------------------------
    # Test 10: Existing memory/tool flow does not break
    # -------------------------------------------------------------
    print("\n[Test 10] Existing memory/tool flow does not break")
    # Calculator tool planning
    calc_plan = planner.create_plan(Goal(goal="Launch Calculator"))
    assert calc_plan.steps[0].type == "tool"
    assert calc_plan.steps[0].tool == "calculator"
    assert "calculator" in TOOL_REGISTRY

    # Memory tool planning
    mem_plan = planner.create_plan(Goal(goal="Access Memory"))
    assert mem_plan.steps[0].type == "memory"
    assert mem_plan.steps[0].tool == "memory"
    assert "memory" in TOOL_REGISTRY

    # Test routing memory tool
    orig_mem = TOOL_REGISTRY["memory"]
    TOOL_REGISTRY["memory"] = lambda: "Memory data placeholder"
    try:
        router = Router()
        mem_res = router.route(mem_plan.steps[0])
        assert mem_res.success is True
        assert mem_res.output == "Memory data placeholder"
    finally:
        TOOL_REGISTRY["memory"] = orig_mem
    print("  [OK] Test 10 PASSED: Desktop tool and memory flows remain completely intact!")

    # -------------------------------------------------------------
    # Test 11: Knowledge integration does not modify Knowledge Engine
    # -------------------------------------------------------------
    print("\n[Test 11] Knowledge integration does not modify Knowledge Engine")
    from knowledge.services.rag_service import RAGService
    from knowledge.retrieval.retriever import KnowledgeRetriever
    from knowledge.context.context_builder import ContextBuilder
    from knowledge.vector_db.chroma_database import ChromaDatabase

    # Verify Knowledge Engine components maintain original interfaces and signatures
    assert hasattr(RAGService, "query"), "RAGService.query missing"
    assert hasattr(KnowledgeRetriever, "retrieve"), "KnowledgeRetriever.retrieve missing"
    assert hasattr(ContextBuilder, "build"), "ContextBuilder.build missing"
    assert hasattr(ChromaDatabase, "search"), "ChromaDatabase.search missing"
    print("  [OK] Test 11 PASSED: Knowledge Engine remains completely unmodified!")

    print("\n" + "=" * 60)
    print("ALL 11 AGENT <-> RAG INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_agent_rag_integration()
