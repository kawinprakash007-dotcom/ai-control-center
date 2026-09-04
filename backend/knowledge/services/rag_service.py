import logging
from typing import Optional, Union

from knowledge.interfaces.rag_service_interface import RAGServiceInterface
from knowledge.interfaces.retriever_interface import RetrieverInterface
from knowledge.interfaces.context_builder_interface import ContextBuilderInterface
from knowledge.models.query import Query
from knowledge.models.retrieval_result import RetrievalResult
from knowledge.models.knowledge_context import KnowledgeContext
from knowledge.retrieval.retriever import KnowledgeRetriever
from knowledge.context.context_builder import ContextBuilder
from knowledge.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class RAGService(RAGServiceInterface):
    """
    RAG Orchestration Service for the Knowledge Engine.

    Coordinates query validation, retrieval via the retrieval layer,
    and context construction via the context builder into a structured,
    LLM-ready KnowledgeContext.

    Separation of Concerns:
    - Does NOT call LLMs, Ollama, or generate answers
    - Does NOT query vector databases directly (delegates to retriever)
    - Does NOT compute embeddings directly (delegates to retriever)
    - Does NOT format chunks or duplicate context builder logic (delegates to builder)
    """

    def __init__(
        self,
        retriever: Optional[Union[RetrievalService, RetrieverInterface]] = None,
        context_builder: Optional[ContextBuilderInterface] = None,
        default_collection: Optional[str] = "linux"
    ):
        """
        Initialize RAGService with dependency injection support.

        Args:
            retriever: Component implementing .retrieve(query). Defaults to KnowledgeRetriever().
            context_builder: Component implementing .build(retrieval_result). Defaults to ContextBuilder().
            default_collection: Fallback collection name if unspecified in query. Defaults to 'linux'.
        """
        self.retriever = retriever if retriever is not None else KnowledgeRetriever()
        self.context_builder = context_builder if context_builder is not None else ContextBuilder()
        self.default_collection = default_collection

    def query(
        self,
        query: Union[Query, str, None]
    ) -> KnowledgeContext:
        """
        Orchestrate retrieval and context building for a given query.

        Flow:
            Query -> Validation -> Retriever -> RetrievalResult -> ContextBuilder -> KnowledgeContext

        Args:
            query: Query instance or plain text query string.

        Returns:
            Structured KnowledgeContext instance.
        """
        # 1. Validate query input safely without ChromaDB access
        if query is None:
            return KnowledgeContext(query="", chunks=[], formatted_context="")

        if isinstance(query, str):
            query_text = query.strip()
            if not query_text:
                return KnowledgeContext(query="", chunks=[], formatted_context="")
            # Construct internal query object without mutating user input
            active_query = Query(
                text=query_text,
                collection=self.default_collection
            )
            original_query_text = query_text
        elif isinstance(query, Query):
            query_text = query.text.strip() if query.text else ""
            if not query_text:
                return KnowledgeContext(
                    query=query.text if query.text is not None else "",
                    chunks=[],
                    formatted_context=""
                )
            # Create a safe copy to prevent mutating the original Query instance
            collection_to_use = query.collection if query.collection else self.default_collection
            active_query = Query(
                text=query.text,
                collection=collection_to_use,
                top_k=query.top_k,
                filters=query.filters
            )
            original_query_text = query.text
        else:
            return KnowledgeContext(query=str(query), chunks=[], formatted_context="")

        # 2. Invoke retrieval layer safely
        try:
            retrieval_result = self.retriever.retrieve(active_query)
        except Exception as e:
            logger.error("Retrieval failed during RAG orchestration: %s", e)
            return KnowledgeContext(
                query=original_query_text,
                chunks=[],
                formatted_context=""
            )

        # 3. Handle empty or None retrieval result gracefully
        if retrieval_result is None:
            return KnowledgeContext(
                query=original_query_text,
                chunks=[],
                formatted_context=""
            )

        # 4. Invoke context builder safely
        try:
            context = self.context_builder.build(retrieval_result)
        except Exception as e:
            logger.error("Context building failed during RAG orchestration: %s", e)
            return KnowledgeContext(
                query=original_query_text,
                chunks=[],
                formatted_context=""
            )

        if context is None:
            return KnowledgeContext(
                query=original_query_text,
                chunks=[],
                formatted_context=""
            )

        # Ensure original query text is faithfully preserved in output context
        if not context.query and original_query_text:
            context.query = original_query_text

        return context

    # Convenience aliases
    def retrieve_context(self, query: Union[Query, str, None]) -> KnowledgeContext:
        """Alias for query()."""
        return self.query(query)

    def process(self, query: Union[Query, str, None]) -> KnowledgeContext:
        """Alias for query()."""
        return self.query(query)

    def retrieve(self, query: Union[Query, str, None]) -> KnowledgeContext:
        """Alias for query()."""
        return self.query(query)

    def __call__(self, query: Union[Query, str, None]) -> KnowledgeContext:
        """Allow calling service instance directly."""
        return self.query(query)
