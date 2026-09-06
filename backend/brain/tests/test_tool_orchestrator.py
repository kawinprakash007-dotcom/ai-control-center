"""Tests for Phase 3.2 General Tool Orchestrator."""

import pytest
from unittest.mock import MagicMock, patch

from core.models.tool_call import ToolCall
from core.models.result import Result
from core.models.task import Task
from core.models.web import SearchResult, FetchResult, EvidenceSet, EvidenceItem, ResearchResult
from tools.tool_orchestrator import ToolOrchestrator
from tools.capability_registry import CapabilityRegistry
from tools.executor import Executor
from tools.memory_capability import MemoryCapability
from tools.knowledge_capability import KnowledgeCapability
from tools.web_capability import WebCapability
from brain.router import Router


class FakeWebCapability:
    """Mock WebCapability for orchestrator tests."""

    def __init__(self, output="Search results", data=None):
        self.output = output
        self.data = data
        self.calls = []
        self.last_evidence = data if isinstance(data, EvidenceSet) else None
        self.last_research_result = data if isinstance(data, ResearchResult) else None

    def __call__(self, task=None):
        self.calls.append(task)
        return self.output


class FakeMemoryCapability:
    """Mock MemoryCapability for orchestrator tests."""

    def __init__(self, output="Memory operation successful"):
        self.output = output
        self.calls = []

    def __call__(self, task=None):
        self.calls.append(task)
        return self.output


class FakeKnowledgeCapability:
    """Mock KnowledgeCapability for orchestrator tests."""

    def __init__(self, output="Knowledge context excerpt"):
        self.output = output
        self.calls = []

    def __call__(self, task=None):
        self.calls.append(task)
        return self.output


class TestToolCallValidation:
    """Test ToolCall model and parameter validation."""

    def test_valid_tool_call_instantiation(self):
        """1. Valid ToolCall instantiates cleanly with normalized attributes."""
        tc = ToolCall(
            capability="WEB",
            action="SEARCH",
            parameters={"query": "Raspberry Pi 5 benchmarks"},
            reason="Investigate hardware capabilities",
            call_id="call-101",
        )
        assert tc.capability == "web"
        assert tc.action == "search"
        assert tc.parameters["query"] == "Raspberry Pi 5 benchmarks"
        assert tc.reason == "Investigate hardware capabilities"
        assert tc.call_id == "call-101"

    def test_unknown_capability_rejected(self):
        """2. Unregistered or unwhitelisted capability is rejected."""
        orchestrator = ToolOrchestrator()
        tc = ToolCall(
            capability="crypto_miner",
            action="mine",
            parameters={"threads": 4},
        )
        result = orchestrator.execute(tc)
        assert result.success is False
        assert "Unknown or unauthorized capability" in result.message
        assert "crypto_miner" in result.message

    def test_unknown_action_rejected(self):
        """3. Whitelisted capability with unauthorized action is rejected."""
        orchestrator = ToolOrchestrator()
        tc = ToolCall(
            capability="web",
            action="drop_database",
            parameters={"table": "users"},
        )
        result = orchestrator.execute(tc)
        assert result.success is False
        assert "Action 'drop_database' is not permitted for capability 'web'" in result.message

    def test_malformed_parameters_rejected(self):
        """4. Malformed or missing parameters are rejected by validator."""
        orchestrator = ToolOrchestrator()

        # Web search with empty query
        tc_empty_query = ToolCall(capability="web", action="search", parameters={"query": "  "})
        res = orchestrator.execute(tc_empty_query)
        assert res.success is False
        assert "requires a non-empty 'query'" in res.message

        # Web fetch with non-HTTP URL
        tc_bad_url = ToolCall(capability="web", action="fetch", parameters={"url": "ftp://files.org/data"})
        res2 = orchestrator.execute(tc_bad_url)
        assert res2.success is False
        assert "valid HTTP/HTTPS URL" in res2.message

        # Memory save with missing value
        tc_missing_val = ToolCall(capability="memory", action="save", parameters={"key": "theme"})
        res3 = orchestrator.execute(tc_missing_val)
        assert res3.success is False
        assert "requires a non-empty 'value'" in res3.message


class TestCapabilityDispatch:
    """Test dispatching ToolCalls to whitelisted capabilities."""

    def test_memory_dispatch(self):
        """5. Memory capability receives validated ToolCall."""
        mock_mem = FakeMemoryCapability(output="Got it — I'll remember that your theme is dark.")
        reg = CapabilityRegistry(capabilities={"memory": mock_mem})
        orchestrator = ToolOrchestrator(registry=reg)

        tc = ToolCall(
            capability="memory",
            action="save",
            parameters={"key": "theme", "value": "dark"},
            call_id="mem-1",
        )
        result = orchestrator.execute(tc)
        assert result.success is True
        assert "Got it — I'll remember" in str(result.output)
        assert result.capability == "memory"
        assert result.action == "save"
        assert result.call_id == "mem-1"

    def test_knowledge_dispatch(self):
        """6. Knowledge capability receives validated ToolCall."""
        mock_know = FakeKnowledgeCapability(output="Project architecture documentation excerpt.")
        reg = CapabilityRegistry(capabilities={"knowledge": mock_know})
        orchestrator = ToolOrchestrator(registry=reg)

        tc = ToolCall(
            capability="knowledge",
            action="query",
            parameters={"query": "system architecture"},
            call_id="know-1",
        )
        result = orchestrator.execute(tc)
        assert result.success is True
        assert "Project architecture documentation" in str(result.output)
        assert result.capability == "knowledge"
        assert result.action == "query"

    def test_web_search_dispatch(self):
        """7. Web search action dispatches through orchestrator."""
        mock_web = FakeWebCapability(output="Web search results for 'AI chips'")
        reg = CapabilityRegistry(capabilities={"web": mock_web})
        orchestrator = ToolOrchestrator(registry=reg)

        tc = ToolCall(
            capability="web",
            action="search",
            parameters={"query": "AI chips", "max_results": 3},
            call_id="web-search-1",
        )
        result = orchestrator.execute(tc)
        assert result.success is True
        assert "Web search results" in str(result.output)
        assert result.capability == "web"
        assert result.action == "search"
        assert result.call_id == "web-search-1"

    def test_web_fetch_dispatch(self):
        """8. Web fetch action dispatches through orchestrator."""
        mock_web = FakeWebCapability(output="Detailed content from https://example.com/page")
        reg = CapabilityRegistry(capabilities={"web": mock_web})
        orchestrator = ToolOrchestrator(registry=reg)

        tc = ToolCall(
            capability="web",
            action="fetch",
            parameters={"url": "https://example.com/page"},
        )
        result = orchestrator.execute(tc)
        assert result.success is True
        assert "Detailed content" in str(result.output)
        assert result.capability == "web"
        assert result.action == "fetch"

    def test_web_research_dispatch(self):
        """9. Web research action dispatches through orchestrator."""
        mock_web = FakeWebCapability(output="Synthesized research report")
        reg = CapabilityRegistry(capabilities={"web": mock_web})
        orchestrator = ToolOrchestrator(registry=reg)

        tc = ToolCall(
            capability="web",
            action="research",
            parameters={"query": "quantum computing breakthroughs 2026", "max_iterations": 2},
        )
        result = orchestrator.execute(tc)
        assert result.success is True
        assert "Synthesized research report" in str(result.output)
        assert result.capability == "web"
        assert result.action == "research"


class TestOrchestratorSafetyAndIsolation:
    """Test safety constraints, dependency injection, and isolation."""

    def test_execution_failure_handling(self):
        """10. Capability runtime exceptions are cleanly trapped into failed Results."""
        def failing_capability(task):
            raise RuntimeError("Database connection timed out")

        reg = CapabilityRegistry(capabilities={"memory": failing_capability})
        orchestrator = ToolOrchestrator(registry=reg)

        tc = ToolCall(capability="memory", action="read", parameters={"key": "user_id"})
        result = orchestrator.execute(tc)
        assert result.success is False
        assert "Database connection timed out" in result.message

    def test_injected_fake_registry(self):
        """11. Orchestrator operates seamlessly with an injected fake registry."""
        fake_reg = CapabilityRegistry(capabilities={
            "web": FakeWebCapability(output="isolated test web")
        })
        orchestrator = ToolOrchestrator(registry=fake_reg)
        res = orchestrator.execute(ToolCall(capability="web", action="search", parameters={"query": "test"}))
        assert res.success is True
        assert res.output == "isolated test web"

    def test_injected_fake_capability(self):
        """12. Custom fake capability can be dynamically registered."""
        reg = CapabilityRegistry(capabilities={})
        reg.register("web", FakeWebCapability(output="dynamic fake"))
        orchestrator = ToolOrchestrator(registry=reg)
        res = orchestrator.execute(ToolCall(capability="web", action="search", parameters={"query": "test"}))
        assert res.success is True
        assert res.output == "dynamic fake"

    def test_structured_data_preservation(self):
        """13. Structured Result.data (e.g. EvidenceSet) is preserved during orchestration."""
        item = EvidenceItem(
            id="ev-10",
            title="Benchmark",
            url="https://example.com/benchmark",
            domain="example.com",
            content="Benchmark numbers",
            retrieved_at="2026-09-06T12:00:00Z",
        )
        ev_set = EvidenceSet(items=(item,))
        mock_web = FakeWebCapability(output="Summary", data=ev_set)
        reg = CapabilityRegistry(capabilities={"web": mock_web})
        orchestrator = ToolOrchestrator(registry=reg)

        res = orchestrator.execute(ToolCall(capability="web", action="search", parameters={"query": "benchmarks"}))
        assert res.success is True
        assert res.data is not None
        assert isinstance(res.data, EvidenceSet)
        assert len(res.data) == 1

    def test_call_id_propagation(self):
        """14. call_id propagates through Result for observability."""
        reg = CapabilityRegistry(capabilities={"web": FakeWebCapability()})
        orchestrator = ToolOrchestrator(registry=reg)

        tc = ToolCall(
            capability="web",
            action="search",
            parameters={"query": "telemetry test"},
            call_id="trace-abc-123",
        )
        result = orchestrator.execute(tc)
        assert result.call_id == "trace-abc-123"

    def test_arbitrary_callable_execution_blocked(self):
        """15. Calling arbitrary Python functions or unapproved capabilities is blocked."""
        orchestrator = ToolOrchestrator()
        tc = ToolCall(
            capability="open_calculator",
            action="run",
            parameters={},
        )
        result = orchestrator.execute(tc)
        assert result.success is False
        assert "Unknown or unauthorized capability" in result.message

    def test_shell_execution_blocked(self):
        """16. Shell / command execution is strictly blocked."""
        orchestrator = ToolOrchestrator()
        tc = ToolCall(
            capability="shell",
            action="bash",
            parameters={"cmd": "cat /etc/passwd"},
        )
        result = orchestrator.execute(tc)
        assert result.success is False
        assert ("Prohibited" in result.message) or ("unauthorized" in result.message)

    def test_filesystem_execution_blocked(self):
        """17. Filesystem manipulation is strictly blocked."""
        orchestrator = ToolOrchestrator()
        tc = ToolCall(
            capability="filesystem",
            action="delete_file",
            parameters={"path": "/important.txt"},
        )
        result = orchestrator.execute(tc)
        assert result.success is False
        assert ("Prohibited" in result.message) or ("unauthorized" in result.message)


class TestRouterAndExecutorCompatibility:
    """Test compatibility with legacy Router, Executor, and Task."""

    def test_router_delegates_to_orchestrator_for_allowed_capability(self):
        """18 & 19. Router routes allowed capabilities through ToolOrchestrator."""
        mock_web = FakeWebCapability(output="Web routed through orchestrator")
        reg = CapabilityRegistry(capabilities={"web": mock_web})
        orchestrator = ToolOrchestrator(registry=reg)
        router = Router(orchestrator=orchestrator)

        task = Task(
            id=42,
            type="web",
            action="search",
            tool="web",
            parameters={"action": "search", "query": "python concurrency"},
        )
        result = router.route(task)
        assert isinstance(result, Result)
        assert result.success is True
        assert result.output == "Web routed through orchestrator"
        assert result.capability == "web"
        assert result.call_id == "42"

    def test_router_direct_tool_call_routing(self):
        """Router directly routes ToolCall objects."""
        mock_mem = FakeMemoryCapability(output="Memory routed")
        reg = CapabilityRegistry(capabilities={"memory": mock_mem})
        orchestrator = ToolOrchestrator(registry=reg)
        router = Router(orchestrator=orchestrator)

        tc = ToolCall(capability="memory", action="read", parameters={})
        result = router.route(tc)
        assert result.success is True
        assert result.output == "Memory routed"

    def test_executor_task_compatibility(self):
        """20. Standard Executor executes capabilities without regression."""
        executor = Executor()
        mock_web = FakeWebCapability(output="Executor output")
        task = Task(id=1, type="web", action="search", tool="web", parameters={"query": "test"})
        res = executor.execute(mock_web, task)
        assert res.success is True
        assert res.output == "Executor output"

    def test_dict_input_to_orchestrator(self):
        """Orchestrator accepts dict inputs cleanly."""
        mock_know = FakeKnowledgeCapability(output="Dictionary call")
        reg = CapabilityRegistry(capabilities={"knowledge": mock_know})
        orchestrator = ToolOrchestrator(registry=reg)

        dict_call = {
            "capability": "knowledge",
            "action": "query",
            "parameters": {"query": "sample"},
            "call_id": "dict-1",
        }
        res = orchestrator.execute(dict_call)
        assert res.success is True
        assert res.output == "Dictionary call"
        assert res.call_id == "dict-1"
