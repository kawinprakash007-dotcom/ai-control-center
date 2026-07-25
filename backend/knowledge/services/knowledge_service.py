from knowledge.models.query import Query
from knowledge.models.knowledge_context import KnowledgeContext

from knowledge.services.retrieval_service import RetrievalService
from knowledge.services.indexing_service import IndexingService


class KnowledgeService:

    """
    Main entry point of the Knowledge Engine.
    """

    def __init__(

        self,

        retrieval_service: RetrievalService,

        indexing_service: IndexingService

    ):

        self.retrieval_service = retrieval_service

        self.indexing_service = indexing_service

    def ask(

        self,

        query: Query

    ) -> KnowledgeContext:

        raise NotImplementedError(
            "Knowledge retrieval is not implemented yet."
        )

    def add_document(

        self,

        path: str

    ):

        raise NotImplementedError(
            "Document ingestion is not implemented yet."
        )