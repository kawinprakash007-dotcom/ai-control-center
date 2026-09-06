import pytest
from datetime import datetime
from typing import List

from core.models.web import (
    SearchResult,
    FetchResult,
    EvidenceItem,
    EvidenceSet,
    create_evidence_from_search,
    create_evidence_from_fetch,
    normalize_domain,
    canonicalize_url,
)
from core.models.result import Result
from core.models.task import Task
from core.models.plan import Plan
from core.interfaces.web_interface import WebProviderInterface, WebProviderError
from tools.web_capability import WebCapability
from tools.executor import Executor
from brain.router import Router
from brain.verification import StandardVerifier


# ============================================================================
# FAKE PROVIDER FOR DETERMINISTIC TESTING
# ============================================================================

class FakeWebProvider(WebProviderInterface):
    """Deterministic, in-memory web provider for evidence testing."""

    def __init__(
        self,
        canned_search: List[SearchResult] = None,
        canned_fetch: FetchResult = None,
        should_fail: bool = False,
    ):
        self.canned_search = canned_search if canned_search is not None else [
            SearchResult(
                title="Python 3.14 Documentation",
                url="https://docs.python.org/3.14/",
                snippet="Official Python 3.14 documentation and releases.",
                source="docs.python.org",
            ),
            SearchResult(
                title="Python 3.14 Release Schedule",
                url="https://peps.python.org/pep-0745/",
                snippet="PEP 745 timeline and milestones.",
                source="peps.python.org",
            ),
        ]
        self.canned_fetch = canned_fetch or FetchResult(
            url="https://example.com/info",
            title="Example Information",
            content="Detailed text content from the example domain.",
            status_code=200,
            source="example.com",
        )
        self.should_fail = should_fail

    def search(self, query: str, max_results: int = 5, timeout_seconds: float = 10.0) -> List[SearchResult]:
        if self.should_fail:
            raise WebProviderError("Provider search failure")
        return list(self.canned_search[:max_results])

    def fetch(self, url: str, timeout_seconds: float = 10.0) -> FetchResult:
        if self.should_fail:
            raise WebProviderError("Provider fetch failure")
        return self.canned_fetch


# ============================================================================
# UNIT TESTS: EVIDENCE MODELS & CONVERTERS
# ============================================================================

def test_search_result_to_evidence_item():
    """1. SearchResult converts cleanly to EvidenceItem."""
    sr = SearchResult(
        title="Python 3.14 Notes",
        url="https://docs.python.org/3.14/notes",
        snippet="Summary of Python 3.14 features.",
        source="docs.python.org",
    )
    ev = create_evidence_from_search(sr)

    assert isinstance(ev, EvidenceItem)
    assert ev.title == "Python 3.14 Notes"
    assert ev.url == "https://docs.python.org/3.14/notes"
    assert ev.content == "Summary of Python 3.14 features."
    assert ev.domain == "docs.python.org"
    assert ev.id.startswith("ev-")
    assert ev.metadata.get("type") == "search_result"


def test_fetch_result_to_evidence_item():
    """2. FetchResult converts cleanly to EvidenceItem."""
    fr = FetchResult(
        url="https://example.com/article",
        title="Sample Article",
        content="This is the full fetched article content.",
        status_code=200,
        source="example.com",
    )
    ev = create_evidence_from_fetch(fr)

    assert isinstance(ev, EvidenceItem)
    assert ev.title == "Sample Article"
    assert ev.url == "https://example.com/article"
    assert ev.content == "This is the full fetched article content."
    assert ev.domain == "example.com"
    assert ev.metadata.get("status_code") == 200
    assert ev.metadata.get("type") == "fetch_result"


def test_url_preservation():
    """3. Exact target URL is preserved without destructive rewriting."""
    exact_url = "https://sub.domain.org:8443/deep/path/page.html?arg=1&foo=Bar#frag"
    sr = SearchResult(title="Test", url=exact_url, snippet="Snippet")
    ev = create_evidence_from_search(sr)

    assert ev.url == exact_url


def test_domain_extraction():
    """4. Domain extraction normalizes www and ports cleanly."""
    assert normalize_domain("https://www.example.com/page") == "example.com"
    assert normalize_domain("https://api.github.com:443/repos") == "api.github.com"
    assert normalize_domain("http://localhost:8000/status") == "localhost"
    assert normalize_domain("", fallback_source="duckduckgo") == "duckduckgo"


def test_retrieved_at_population():
    """5. retrieved_at is populated with a valid timestamp."""
    sr = SearchResult(title="T", url="https://example.com", snippet="S")
    ev1 = create_evidence_from_search(sr)
    assert ev1.retrieved_at is not None
    assert len(ev1.retrieved_at) > 10

    custom_ts = "2026-09-06T12:00:00Z"
    ev2 = create_evidence_from_search(sr, retrieved_at=custom_ts)
    assert ev2.retrieved_at == custom_ts


def test_snippet_preservation():
    """6. Snippet text is completely preserved in EvidenceItem.content."""
    snippet_text = "Detailed excerpt with special characters: &lt;tag&gt; 'quotes' and € symbols."
    sr = SearchResult(title="T", url="https://example.com", snippet=snippet_text)
    ev = create_evidence_from_search(sr)

    assert ev.content == snippet_text


def test_fetched_content_preservation():
    """7. Cleaned page text is preserved in EvidenceItem.content."""
    cleaned_body = "Paragraph 1\nParagraph 2\nKey findings: accuracy=98.5%."
    fr = FetchResult(url="https://example.com", title="T", content=cleaned_body, status_code=200)
    ev = create_evidence_from_fetch(fr)

    assert ev.content == cleaned_body


def test_status_code_preservation():
    """8. HTTP status code is stored in metadata."""
    fr = FetchResult(url="https://example.com", title="T", content="Content", status_code=206)
    ev = create_evidence_from_fetch(fr)

    assert ev.metadata["status_code"] == 206


def test_deterministic_ordering_in_evidence_set():
    """9. EvidenceSet maintains the original insertion order of items."""
    items = [
        EvidenceItem(id="ev-1", title="First", url="https://a.com/1", domain="a.com", content="C1", retrieved_at="T1"),
        EvidenceItem(id="ev-2", title="Second", url="https://b.com/2", domain="b.com", content="C2", retrieved_at="T2"),
        EvidenceItem(id="ev-3", title="Third", url="https://c.com/3", domain="c.com", content="C3", retrieved_at="T3"),
    ]
    ev_set = EvidenceSet.from_items(items)

    assert len(ev_set) == 3
    assert [item.title for item in ev_set] == ["First", "Second", "Third"]
    assert ev_set[0].id == "ev-1"
    assert ev_set[1].id == "ev-2"
    assert ev_set[2].id == "ev-3"


def test_duplicate_url_handling():
    """10. Duplicate canonical URLs are deduplicated, keeping the first occurrence."""
    items = [
        EvidenceItem(id="ev-1", title="Original", url="https://example.com/page", domain="example.com", content="Orig", retrieved_at="T1"),
        EvidenceItem(id="ev-2", title="Duplicate Slash", url="https://example.com/page/", domain="example.com", content="Dup", retrieved_at="T2"),
        EvidenceItem(id="ev-3", title="Duplicate WWW", url="https://www.example.com/page", domain="example.com", content="Dup", retrieved_at="T3"),
    ]
    ev_set = EvidenceSet.from_items(items)

    assert len(ev_set) == 1
    assert ev_set[0].title == "Original"


def test_distinct_urls_on_same_domain_remain_distinct():
    """11. Different paths on the same domain are NOT deduplicated."""
    items = [
        EvidenceItem(id="ev-1", title="Page A", url="https://example.com/a", domain="example.com", content="A", retrieved_at="T1"),
        EvidenceItem(id="ev-2", title="Page B", url="https://example.com/b", domain="example.com", content="B", retrieved_at="T2"),
        EvidenceItem(id="ev-3", title="Page C", url="https://example.com/c", domain="example.com", content="C", retrieved_at="T3"),
    ]
    ev_set = EvidenceSet.from_items(items)

    assert len(ev_set) == 3
    assert [item.title for item in ev_set] == ["Page A", "Page B", "Page C"]


def test_empty_search_results():
    """12. Zero search results produce an empty EvidenceSet."""
    fake_provider = FakeWebProvider(canned_search=[])
    capability = WebCapability(provider=fake_provider)

    output = capability({"action": "search", "query": "nonexistent query xyz"})

    assert "No web search results found" in output
    assert capability.last_evidence is not None
    assert capability.last_evidence.is_empty
    assert len(capability.last_evidence) == 0


def test_failed_fetch_creates_no_false_evidence():
    """13. Failed fetch leaves last_evidence as None with no false evidence."""
    failing_provider = FakeWebProvider(should_fail=True)
    capability = WebCapability(provider=failing_provider)

    with pytest.raises(WebProviderError):
        capability({"action": "fetch", "url": "https://broken.example.com"})

    assert capability.last_evidence is None


def test_evidence_serialization():
    """14. EvidenceItem and EvidenceSet round-trip safely to and from dict."""
    item = EvidenceItem(
        id="ev-test1",
        title="Serialization Test",
        url="https://example.com/test",
        domain="example.com",
        content="Test content for serialization.",
        retrieved_at="2026-09-06T12:00:00Z",
        metadata={"status_code": 200, "source": "example.com"},
    )
    d = item.to_dict()
    restored = EvidenceItem.from_dict(d)
    assert restored == item

    ev_set = EvidenceSet.from_items([item], query="test query", max_items=10)
    set_dict = ev_set.to_dict()
    restored_set = EvidenceSet.from_dict(set_dict)
    assert len(restored_set) == 1
    assert restored_set[0] == item
    assert restored_set.query == "test query"


def test_evidence_set_bounds():
    """15. EvidenceSet bounds item count strictly to max_items."""
    items = [
        EvidenceItem(id=f"ev-{i}", title=f"Title {i}", url=f"https://example.com/{i}", domain="example.com", content=f"C{i}", retrieved_at="T")
        for i in range(20)
    ]
    ev_set = EvidenceSet.from_items(items, max_items=5)

    assert len(ev_set) == 5
    assert ev_set.max_items == 5
    assert [item.title for item in ev_set] == [f"Title {i}" for i in range(5)]


# ============================================================================
# INTEGRATION TESTS: CAPABILITY, EXECUTOR, & RESULT INTEGRATION
# ============================================================================

def test_web_capability_exposes_evidence_correctly():
    """16. WebCapability populates last_evidence and task.parameters['evidence']."""
    fake_provider = FakeWebProvider()
    capability = WebCapability(provider=fake_provider)

    task = Task(
        id=1,
        type="web",
        action="Web Search",
        tool="web",
        parameters={"action": "search", "query": "Python 3.14"},
        status="pending",
    )

    output = capability(task)

    assert "Python 3.14 Documentation" in output
    assert capability.last_evidence is not None
    assert len(capability.last_evidence) == 2
    assert task.parameters.get("evidence") == capability.last_evidence
    assert capability.get_evidence() == capability.last_evidence


def test_existing_result_behavior_remains_compatible():
    """17. Result transports evidence in data while keeping output and status backward compatible."""
    fake_provider = FakeWebProvider()
    capability = WebCapability(provider=fake_provider)

    executor = Executor()
    task = Task(
        id=1,
        type="web",
        action="Web Search",
        tool="web",
        parameters={"action": "search", "query": "Python release"},
        status="pending",
    )

    res = executor.execute(capability, task)

    assert isinstance(res, Result)
    assert res.success is True
    assert res.message == "Task completed."
    assert isinstance(res.output, str)
    assert "Python 3.14 Documentation" in res.output
    # Structured evidence is attached to Result.data
    assert isinstance(res.data, EvidenceSet)
    assert len(res.data) == 2
    assert res.data[0].domain == "docs.python.org"


def test_verification_remains_correct_with_evidence():
    """18. StandardVerifier confirms success when evidence is present."""
    verifier = StandardVerifier()
    task = Task(
        id=1,
        type="web",
        action="Web Search",
        tool="web",
        status="completed",
    )
    plan = Plan(goal="web_search", steps=[task], status="completed")
    results = [Result(success=True, message="Task completed.", output="Formatted string", data=EvidenceSet())]

    ver_res = verifier.verify(plan, results)
    assert ver_res.verified is True
    assert ver_res.status == "verified"
