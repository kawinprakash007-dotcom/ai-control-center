from abc import ABC, abstractmethod
from typing import Optional, Union

from knowledge.models.query import Query
from knowledge.models.knowledge_context import KnowledgeContext


class RAGServiceInterface(ABC):
    """
    Interface for RAG orchestration services.
    Coordinates retrieval and context construction without executing LLM generation.
    """

    @abstractmethod
    def query(
        self,
        query: Union[Query, str, None]
    ) -> KnowledgeContext:
        """
        Orchestrate retrieval and context construction for a given query.

        Args:
            query: Query instance or plain text query string.

        Returns:
            KnowledgeContext containing preserved query, selected chunks, and formatted text.
        """
        pass
