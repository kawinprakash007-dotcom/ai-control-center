from abc import ABC, abstractmethod

from knowledge.models.embedding import Embedding
from knowledge.models.query import Query
from knowledge.models.retrieval_result import RetrievalResult


class VectorDatabaseInterface(ABC):

    @abstractmethod
    def create_collection(
        self,
        name: str
    ):
        pass

    @abstractmethod
    def add_embedding(
        self,
        collection: str,
        embedding: Embedding
    ):
        pass

    @abstractmethod
    def search(
        self,
        query: Query
    ) -> RetrievalResult:
        pass

    @abstractmethod
    def delete_collection(
        self,
        name: str
    ):
        pass