import inspect
from unittest.mock import MagicMock, patch
import pytest

from core.interfaces.web_interface import WebProviderInterface, WebProviderError
from core.models.web import SearchResult, FetchResult
from core.models.task import Task
from core.models.result import Result
from tools.web_capability import WebCapability
from tools.capability_registry import CapabilityRegistry
from tools.executor import Executor
from brain.router import Router
from web.default_provider import DefaultWebProvider


# ============================================================================
# FAKES & TEST DOUBLES
# ============================================================================

class FakeWebProvider(WebProviderInterface):
    """Deterministic in-memory fake implementing WebProviderInterface."""

    def __init__(self, search_results=None, fetch_result=None, should_fail=False):
        self.search_results = search_results if search_results is not None else [
            SearchResult(
                title="Python Official",
                url="https://www.python.org",
                snippet="Python programming language official site.",
                source="python.org",
            ),
            SearchResult(
                title="Python Tutorial",
                url="https://docs.python.org/3/tutorial/",
                snippet="Official Python tutorial documentation.",
                source="docs.python.org",
            ),
        ]
        self.fetch_result = fetch_result if fetch_result is not None else FetchResult(
            url="https://example.com",
            title="Example Domain",
            content="This domain is established to be used for illustrative examples.",
            status_code=200,
            source="example.com",
        )
        self.should_fail = should_fail
        self.last_search_query = None
        self.last_fetch_url = None

    def search(self, query: str, max_results: int = 5, timeout_seconds: float = 10.0):
        self.last_search_query = query
        if self.should_fail:
            raise WebProviderError("Simulated network search failure")
        return self.search_results[:max_results]

    def fetch(self, url: str, timeout_seconds: float = 10.0):
        self.last_fetch_url = url
        if self.should_fail:
            raise WebProviderError("Simulated network fetch failure")
        return self.fetch_result


# ============================================================================
# 1. WEBCAPABILITY SEARCH & FETCH
# ============================================================================

def test_web_capability_search():
    """1. WebCapability executes search action and returns formatted results."""
    provider = FakeWebProvider()
    cap = WebCapability(provider=provider)

    task = Task(
        id=1,
        type="web",
        action="Web Search",
        tool="web",
        parameters={"action": "search", "query": "python"},
    )

    output = cap(task)
    assert "Web search results for 'python':" in output
    assert "1. Python Official" in output
    assert "URL: https://www.python.org" in output
    assert "Snippet: Python programming language official site." in output
    assert provider.last_search_query == "python"


def test_web_capability_fetch():
    """2. WebCapability executes fetch action and returns formatted content."""
    provider = FakeWebProvider()
    cap = WebCapability(provider=provider)

    task = Task(
        id=2,
        type="web",
        action="Web Fetch",
        tool="web",
        parameters={"action": "fetch", "url": "https://example.com"},
    )

    output = cap(task)
    assert "URL: https://example.com" in output
    assert "Title: Example Domain" in output
    assert "Source: example.com" in output
    assert "This domain is established to be used for illustrative examples." in output
    assert provider.last_fetch_url == "https://example.com"


# ============================================================================
# 2. VALIDATION & ERROR HANDLING
# ============================================================================

def test_missing_search_query():
    """3. Missing query for search raises ValueError and produces failed Result."""
    cap = WebCapability(provider=FakeWebProvider())
    executor = Executor()

    task = Task(
        id=1,
        type="web",
        action="Web Search",
        tool="web",
        parameters={"action": "search", "query": ""},
    )

    result = executor.execute(cap, task)
    assert result.success is False
    assert "Missing query for web search" in result.message


def test_missing_fetch_url():
    """4. Missing URL for fetch raises ValueError and produces failed Result."""
    cap = WebCapability(provider=FakeWebProvider())
    executor = Executor()

    task = Task(
        id=1,
        type="web",
        action="Web Fetch",
        tool="web",
        parameters={"action": "fetch", "url": ""},
    )

    result = executor.execute(cap, task)
    assert result.success is False
    assert "Missing URL for web fetch" in result.message


def test_unknown_action():
    """5. Unknown or missing action raises ValueError and produces failed Result."""
    cap = WebCapability(provider=FakeWebProvider())
    executor = Executor()

    task_unknown = Task(
        id=1,
        type="web",
        action="Web Unknown",
        tool="web",
        parameters={"action": "scrape", "url": "https://example.com"},
    )
    result_unknown = executor.execute(cap, task_unknown)
    assert result_unknown.success is False
    assert "Unsupported web action: 'scrape'" in result_unknown.message

    task_empty = Task(
        id=2,
        type="web",
        action="Web Empty",
        tool="web",
        parameters={},
    )
    result_empty = executor.execute(cap, task_empty)
    assert result_empty.success is False
    assert "No web action specified" in result_empty.message


def test_provider_failure():
    """6. Provider network failures result in failed Result rather than unhandled crash."""
    failing_provider = FakeWebProvider(should_fail=True)
    cap = WebCapability(provider=failing_provider)
    executor = Executor()

    task = Task(
        id=1,
        type="web",
        action="Web Search",
        tool="web",
        parameters={"action": "search", "query": "python"},
    )

    result = executor.execute(cap, task)
    assert result.success is False
    assert "Simulated network search failure" in result.message


# ============================================================================
# 3. METADATA & CONTENT PRESERVATION
# ============================================================================

def test_injected_fake_provider():
    """7. WebCapability accepts and delegates strictly to injected provider."""
    fake = FakeWebProvider()
    cap = WebCapability(provider=fake)
    assert cap.provider is fake


def test_search_result_metadata_preservation():
    """8. SearchResult preserves title, url, snippet, and source domain."""
    sr = SearchResult(
        title="Sample Title",
        url="https://sample.org/item",
        snippet="A brief snippet of content.",
        source="sample.org",
    )
    assert sr.title == "Sample Title"
    assert sr.url == "https://sample.org/item"
    assert sr.snippet == "A brief snippet of content."
    assert sr.source == "sample.org"


def test_fetched_content_preservation():
    """9. FetchResult preserves url, title, content, status code, and source."""
    fr = FetchResult(
        url="https://docs.sample.org",
        title="Documentation",
        content="Clean extracted textual content.",
        status_code=200,
        source="docs.sample.org",
    )
    assert fr.url == "https://docs.sample.org"
    assert fr.title == "Documentation"
    assert fr.content == "Clean extracted textual content."
    assert fr.status_code == 200
    assert fr.source == "docs.sample.org"


def test_timeout_and_error_conversion():
    """10. Timeout parameter is passed to provider and timeout errors convert to failed Result."""
    mock_provider = MagicMock(spec=WebProviderInterface)
    mock_provider.search.side_effect = WebProviderError("Connection timeout after 3.0s")
    cap = WebCapability(provider=mock_provider)
    executor = Executor()

    task = Task(
        id=1,
        type="web",
        action="Web Search",
        tool="web",
        parameters={"action": "search", "query": "test", "timeout_seconds": 3.0},
    )

    result = executor.execute(cap, task)
    assert result.success is False
    assert "Connection timeout after 3.0s" in result.message
    mock_provider.search.assert_called_once_with(query="test", max_results=5, timeout_seconds=3.0)


# ============================================================================
# 4. CAPABILITY REGISTRY & ROUTING INTEGRATION
# ============================================================================

def test_capability_registry_contains_web():
    """11. CapabilityRegistry exposes WebCapability under 'web'."""
    registry = CapabilityRegistry()
    web_cap = registry.get_executor("web")
    assert web_cap is not None
    assert isinstance(web_cap, WebCapability)


def test_existing_knowledge_capability_still_works():
    """12. KnowledgeCapability remains registered and functioning."""
    registry = CapabilityRegistry()
    know_cap = registry.get_executor("knowledge")
    assert know_cap is not None
    # Empty task returns deterministic message
    res = know_cap("")
    assert "No query provided" in res


def test_existing_memory_capability_still_works():
    """13. MemoryCapability remains registered and functioning."""
    registry = CapabilityRegistry()
    mem_cap = registry.get_executor("memory")
    assert mem_cap is not None
    # Empty task returns deterministic message
    res = mem_cap(Task(id=1, type="memory", action="Manage Memory", tool="memory", parameters={}))
    assert "No memory operation specified" in res


def test_task_parameters_reach_web_capability_via_router():
    """14. Router invokes WebCapability with Task parameters correctly."""
    fake_provider = FakeWebProvider()
    custom_cap = WebCapability(provider=fake_provider)

    router = Router()
    # Inject test capability into registry for this test
    with patch.object(router.registry, "get_executor", return_value=custom_cap):
        task = Task(
            id=10,
            type="web",
            action="Search Documentation",
            tool="web",
            parameters={"action": "search", "query": "control center", "max_results": 2},
        )
        result = router.route(task)
        assert isinstance(result, Result)
        assert result.success is True
        assert "Web search results for 'control center':" in result.output
        assert fake_provider.last_search_query == "control center"


# ============================================================================
# 5. INDEPENDENCE & ISOLATION (ZERO UNWANTED DEPENDENCIES)
# ============================================================================

def test_no_direct_sqlite_dependency():
    """15. Web module and capability do not import sqlite3 or SQLiteMemoryStore."""
    import tools.web_capability as wc_mod
    import web.default_provider as wp_mod

    wc_source = inspect.getsource(wc_mod)
    wp_source = inspect.getsource(wp_mod)

    assert "sqlite3" not in wc_source.lower()
    assert "sqlite3" not in wp_source.lower()
    assert "sqlitememorystore" not in wc_source.lower()
    assert "sqlitememorystore" not in wp_source.lower()


def test_no_ollama_dependency():
    """16. Web capability does not import or invoke Ollama client."""
    import tools.web_capability as wc_mod
    import web.default_provider as wp_mod

    wc_source = inspect.getsource(wc_mod)
    wp_source = inspect.getsource(wp_mod)

    assert "ollama" not in wc_source.lower()
    assert "ollama" not in wp_source.lower()


def test_no_rag_dependency():
    """17. Web capability does not import or invoke RAG / Knowledge retrieval."""
    import tools.web_capability as wc_mod
    import web.default_provider as wp_mod

    wc_source = inspect.getsource(wc_mod)
    wp_source = inspect.getsource(wp_mod)

    assert "ragservice" not in wc_source.lower()
    assert "ragservice" not in wp_source.lower()
    assert "retriever" not in wc_source.lower()
    assert "retriever" not in wp_source.lower()


# ============================================================================
# 6. DEFAULT WEB PROVIDER PARSING & ERROR TESTS (WITH MOCKED HTTP)
# ============================================================================

def test_default_web_provider_search_parser():
    """Verify DefaultWebProvider parses search HTML into SearchResult objects."""
    provider = DefaultWebProvider()
    mock_html = """
    <html>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fpage">Example <b>Title</b></a>
        <a class="result__snippet">Example <i>snippet</i> describing the page.</a>
    </html>
    """

    mock_resp = MagicMock()
    mock_resp.text = mock_html
    mock_resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_resp):
        results = provider.search("test query", max_results=5)
        assert len(results) == 1
        assert results[0].title == "Example Title"
        assert results[0].url == "https://example.org/page"
        assert results[0].snippet == "Example snippet describing the page."
        assert results[0].source == "example.org"


def test_default_web_provider_fetch_cleaner():
    """Verify DefaultWebProvider cleans HTML scripts/tags and extracts title/content."""
    provider = DefaultWebProvider()
    mock_html = """
    <html>
        <head>
            <title>Page &amp; Title</title>
            <script>console.log("ignore me");</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <h1>Main Heading</h1>
            <p>First paragraph of useful content.</p>
        </body>
    </html>
    """

    mock_resp = MagicMock()
    mock_resp.text = mock_html
    mock_resp.status_code = 200
    mock_resp.url = "https://example.org/article"
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp):
        res = provider.fetch("https://example.org/article")
        assert res.title == "Page & Title"
        assert res.url == "https://example.org/article"
        assert "console.log" not in res.content
        assert "color: red" not in res.content
        assert "Main Heading" in res.content
        assert "First paragraph of useful content." in res.content
        assert res.source == "example.org"


def test_default_web_provider_empty_and_invalid_inputs():
    """Verify DefaultWebProvider validates inputs strictly."""
    provider = DefaultWebProvider()

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        provider.search("")

    with pytest.raises(ValueError, match="timeout_seconds must be a positive number"):
        provider.search("query", timeout_seconds=0)

    with pytest.raises(ValueError, match="url must be a non-empty string"):
        provider.fetch("")

    with pytest.raises(ValueError, match="url must start with 'http://' or 'https://'"):
        provider.fetch("ftp://example.com")

    with pytest.raises(ValueError, match="timeout_seconds must be a positive number"):
        provider.fetch("https://example.com", timeout_seconds=-1)
