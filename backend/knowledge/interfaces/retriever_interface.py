from abc import ABC, abstractmethod

from knowledge.models.query import Query
from knowledge.models.retrieval_result import RetrievalResult


class RetrieverInterface(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: Query
    ) -> RetrievalResult:
        pass