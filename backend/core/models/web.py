import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class SearchResult:
    """
    Structured result item from a web search operation.

    Attributes:
        title: Title of the search result item.
        url: Target destination URL.
        snippet: Descriptive summary or excerpt.
        source: Originating domain or provider name.
    """
    title: str
    url: str
    snippet: str
    source: str = ""


@dataclass(frozen=True)
class FetchResult:
    """
    Structured result from a web content fetch operation.

    Attributes:
        url: Resolved destination URL.
        title: Extracted page title.
        content: Cleaned textual content extracted from the web page.
        status_code: HTTP response status code.
        source: Domain or origin of the page.
    """
    url: str
    title: str
    content: str
    status_code: int = 200
    source: str = ""


def normalize_domain(url: str, fallback_source: str = "") -> str:
    """
    Extract clean, normalized host/domain from URL or fallback source.
    Strips port numbers and leading 'www.'.

    Examples:
        'https://www.example.com/page' -> 'example.com'
        'https://docs.python.org:443/3/' -> 'docs.python.org'
        '' with fallback 'duckduckgo' -> 'duckduckgo'
    """
    if not url or not isinstance(url, str):
        return (fallback_source or "").strip().lower()

    try:
        parsed = urlparse(url.strip())
        netloc = parsed.netloc or parsed.path
        host = netloc.split(":")[0].strip().lower()
        if host.startswith("www."):
            host = host[4:]
        if host:
            return host
    except Exception:
        pass

    return (fallback_source or "").strip().lower()


def canonicalize_url(url: str) -> str:
    """
    Produce a canonical URL string for deduplication purposes.
    The original URL on EvidenceItem remains intact for citation.
    """
    if not url or not isinstance(url, str):
        return ""
    u = url.strip()
    try:
        parsed = urlparse(u)
        scheme = (parsed.scheme or "https").lower()
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        # Normalize path: strip trailing slash
        path = parsed.path.rstrip("/")
        query = f"?{parsed.query}" if parsed.query else ""
        return f"{scheme}://{netloc}{path}{query}"
    except Exception:
        return u.lower().rstrip("/")


def generate_evidence_id(url: str, title: str = "") -> str:
    """
    Generate a deterministic evidence ID based on canonical URL.
    """
    canonical = canonicalize_url(url) or (title.strip().lower() if title else "unknown")
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"ev-{digest[:16]}"


@dataclass(frozen=True)
class EvidenceItem:
    """
    Immutable, deterministic representation of web-derived evidence.
    Preserves exact source provenance for future reasoning, verification,
    and citation generation.

    Attributes:
        id: Deterministic identifier derived from canonical URL.
        title: Title of the web page or search result.
        url: Exact target destination URL (preserves provenance).
        domain: Normalized domain of the source.
        content: Cleaned textual content or search snippet.
        retrieved_at: ISO 8601 UTC timestamp when the evidence was captured.
        metadata: Transport, HTTP, status, or source metadata.
    """
    id: str
    title: str
    url: str
    domain: str
    content: str
    retrieved_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "content": self.content,
            "retrieved_at": self.retrieved_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceItem":
        return cls(
            id=data["id"],
            title=data.get("title", ""),
            url=data.get("url", ""),
            domain=data.get("domain", ""),
            content=data.get("content", ""),
            retrieved_at=data.get("retrieved_at", ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class EvidenceSet:
    """
    Immutable collection of EvidenceItem records.
    Provides deterministic ordering, deduplication by canonical URL,
    bounded item capacity, and safe serialization.

    Attributes:
        items: Tuple of deduplicated, ordered EvidenceItem instances.
        query: Query or target URL that produced this evidence set.
        max_items: Maximum item capacity boundary.
    """
    items: Tuple[EvidenceItem, ...] = field(default_factory=tuple)
    query: str = ""
    max_items: int = 50

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int) -> EvidenceItem:
        return self.items[index]

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    @classmethod
    def from_items(
        cls,
        items: List[EvidenceItem],
        query: str = "",
        max_items: int = 50,
    ) -> "EvidenceSet":
        """
        Construct EvidenceSet from a list of EvidenceItem objects.
        Applies deterministic deduplication by canonical URL while preserving
        first-seen order and bounding length to max_items.
        """
        deduped: List[EvidenceItem] = []
        seen_keys = set()

        for item in items:
            key = canonicalize_url(item.url) or item.id
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)
            if len(deduped) >= max_items:
                break

        return cls(
            items=tuple(deduped),
            query=query,
            max_items=max_items,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "query": self.query,
            "max_items": self.max_items,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceSet":
        items = [EvidenceItem.from_dict(d) for d in data.get("items", [])]
        return cls(
            items=tuple(items),
            query=data.get("query", ""),
            max_items=data.get("max_items", 50),
        )


def create_evidence_from_search(
    result: SearchResult,
    retrieved_at: Optional[str] = None,
) -> EvidenceItem:
    """
    Convert a raw SearchResult into a structured EvidenceItem preserving provenance.
    """
    timestamp = retrieved_at or datetime.now(timezone.utc).isoformat()
    domain = normalize_domain(result.url, fallback_source=result.source)
    item_id = generate_evidence_id(result.url, title=result.title)

    metadata: Dict[str, Any] = {
        "source": result.source or domain,
        "type": "search_result",
    }

    return EvidenceItem(
        id=item_id,
        title=result.title,
        url=result.url,
        domain=domain,
        content=result.snippet,
        retrieved_at=timestamp,
        metadata=metadata,
    )


def create_evidence_from_fetch(
    result: FetchResult,
    retrieved_at: Optional[str] = None,
) -> EvidenceItem:
    """
    Convert a raw FetchResult into a structured EvidenceItem preserving provenance.
    """
    timestamp = retrieved_at or datetime.now(timezone.utc).isoformat()
    domain = normalize_domain(result.url, fallback_source=result.source)
    item_id = generate_evidence_id(result.url, title=result.title)

    metadata: Dict[str, Any] = {
        "source": result.source or domain,
        "status_code": result.status_code,
        "type": "fetch_result",
    }

    return EvidenceItem(
        id=item_id,
        title=result.title,
        url=result.url,
        domain=domain,
        content=result.content,
        retrieved_at=timestamp,
        metadata=metadata,
    )
