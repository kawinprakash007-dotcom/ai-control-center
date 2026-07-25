from abc import ABC, abstractmethod

from knowledge.models.knowledge_context import KnowledgeContext


class LLMRouterInterface(ABC):

    @abstractmethod
    def generate(
        self,
        context: KnowledgeContext
    ) -> str:
        pass