from abc import ABC, abstractmethod

from knowledge.models.query import Query


class KnowledgeRouterInterface(ABC):

    @abstractmethod
    def select_collection(
        self,
        query: Query
    ) -> list[str]:
        pass