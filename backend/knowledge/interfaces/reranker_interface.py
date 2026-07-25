from abc import ABC, abstractmethod

from knowledge.models.query import Query
from knowledge.models.retrieval_result import RetrievalResult


class RerankerInterface(ABC):

    @abstractmethod
    def rerank(
        self,
        query: Query,
        results: RetrievalResult
    ) -> RetrievalResult:
        pass