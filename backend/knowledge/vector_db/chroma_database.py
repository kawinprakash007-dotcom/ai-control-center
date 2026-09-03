import chromadb

from knowledge.interfaces.vector_database_interface import (
    VectorDatabaseInterface
)

from knowledge.models.embedding import Embedding
from knowledge.models.query import Query
from knowledge.models.retrieval_result import RetrievalResult


class ChromaDatabase(VectorDatabaseInterface):

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path="knowledge/chroma_db"
        )

    def create_collection(
        self,
        name: str
    ):

        return self.client.get_or_create_collection(
            name=name
        )

    def add_embedding(
        self,
        collection: str,
        embedding: Embedding
    ):

        db = self.create_collection(collection)

        metadata = embedding.metadata

        # ChromaDB requires a non-empty metadata dictionary
        if not metadata:

            metadata = {
                "chunk_id": embedding.chunk_id
            }

        db.add(
            ids=[embedding.chunk_id],
            embeddings=[embedding.vector],
            metadatas=[metadata]
        )

    def search(
        self,
        query: Query
    ) -> RetrievalResult:

        raise NotImplementedError(
            "Search will be implemented in the Retriever module."
        )

    def delete_collection(
        self,
        name: str
    ):

        self.client.delete_collection(name)