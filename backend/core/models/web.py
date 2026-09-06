from dataclasses import dataclass


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
