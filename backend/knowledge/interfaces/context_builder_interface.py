from abc import ABC, abstractmethod
from typing import Optional

from knowledge.models.retrieval_result import RetrievalResult
from knowledge.models.knowledge_context import KnowledgeContext


class ContextBuilderInterface(ABC):
    """
    Interface for context building strategies.
    Transforms raw RetrievalResult instances into structured KnowledgeContext.
    """

    @abstractmethod
    def build(
        self,
        retrieval_result: Optional[RetrievalResult]
    ) -> KnowledgeContext:
        """
        Build a structured KnowledgeContext from a RetrievalResult.
        """
        pass
