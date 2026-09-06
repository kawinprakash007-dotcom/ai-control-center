from abc import ABC, abstractmethod
from typing import List

from core.models.web import SearchResult, FetchResult


class WebProviderError(Exception):
    """Raised when a web provider operation fails due to network, parsing, or HTTP errors."""
    pass


class WebProviderInterface(ABC):
    """
    Abstract interface defining the contract for web intelligence providers.
    Provides decoupled search and fetch operations.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 5,
        timeout_seconds: float = 10.0,
    ) -> List[SearchResult]:
        """
        Execute web search for the query and return structured search results.
        """
        pass

    @abstractmethod
    def fetch(
        self,
        url: str,
        timeout_seconds: float = 10.0,
    ) -> FetchResult:
        """
        Fetch web page content from the given URL and return structured content.
        """
        pass
