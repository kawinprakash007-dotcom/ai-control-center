from typing import Optional, Union

from knowledge.interfaces.rag_service_interface import RAGServiceInterface
from knowledge.services.rag_service import RAGService
from core.models.task import Task


class KnowledgeCapability:
    """
    Adapter between AI Control Center's Tool protocol and RAGService.

    Receives task execution requests from Router/Executor, extracts the query,
    delegates retrieval and context building to RAGService, and returns the
    formatted context string for Result.output.
    """

    def __init__(
        self,
        rag_service: Optional[RAGServiceInterface] = None
    ):
        """
        Initialize KnowledgeCapability with injected or default RAGService.

        Args:
            rag_service: Implementation of RAGServiceInterface.
        """
        self.rag_service = rag_service or RAGService()

    def __call__(
        self,
        task: Optional[Union[Task, str]] = None
    ) -> str:
        """
        Execute knowledge retrieval for the given task.

        Args:
            task: Task instance containing parameters['query'], or a raw query string.

        Returns:
            KnowledgeContext.formatted_context string or a safe descriptive message.
        """
        query_text = ""

        if isinstance(task, str):
            query_text = task.strip()
        elif task is not None and hasattr(task, "parameters") and isinstance(task.parameters, dict):
            query_text = str(task.parameters.get("query", "")).strip()

        if not query_text:
            return "No query provided for knowledge retrieval."

        context = self.rag_service.query(query_text)

        if context and context.formatted_context:
            return context.formatted_context

        return "No relevant knowledge found."
