from knowledge.interfaces.retriever_interface import RetrieverInterface
from knowledge.interfaces.vector_database_interface import VectorDatabaseInterface
from knowledge.interfaces.embedder_interface import EmbedderInterface

from knowledge.models.query import Query
from knowledge.models.retrieval_result import RetrievalResult

from knowledge.vector_db.chroma_database import ChromaDatabase
from knowledge.embeddings.embedding_factory import EmbeddingFactory


class KnowledgeRetriever(RetrieverInterface):
    """
    Knowledge Retriever Engine.

    Coordinates query validation, query embedding generation,
    and vector similarity search against the vector database.
    """

    def __init__(
        self,
        vector_database: VectorDatabaseInterface | None = None,
        embedder: EmbedderInterface | None = None
    ):
        self.vector_database = vector_database or ChromaDatabase()
        self.embedder = embedder or EmbeddingFactory.create()

    def retrieve(
        self,
        query: Query
    ) -> RetrievalResult:
        """
        Execute vector similarity retrieval for a given user query.

        Flow:
        Query -> Validation -> Embedding -> Vector DB Search -> RetrievalResult

        Score Semantics:
        - RetrievalResult.scores: List of raw Chroma distances (lower values denote closer matches).
        - chunk.metadata["distance"]: Raw Chroma distance metric (float >= 0.0).
        - chunk.metadata["similarity"]: Normalized similarity score 1 / (1 + distance) in (0, 1]
          (higher values denote closer matches).
        """
        if not query or not query.text or not query.text.strip():
            return RetrievalResult(
                query=query.text if query else "",
                chunks=[],
                scores=[]
            )

        if query.top_k <= 0:
            return RetrievalResult(
                query=query.text,
                chunks=[],
                scores=[]
            )

        # Generate query embedding
        if hasattr(self.embedder, "embed_query"):
            query_vector = self.embedder.embed_query(query.text.strip())
        elif hasattr(self.embedder, "embed_text"):
            query_vector = self.embedder.embed_text(query.text.strip())
        else:
            raise AttributeError(
                "Embedder does not support text/query embedding."
            )

        # Delegate similarity search to vector database
        return self.vector_database.search(
            query=query,
            query_embedding=query_vector
        )
