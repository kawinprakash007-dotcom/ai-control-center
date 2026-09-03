from pathlib import Path
import chromadb

from knowledge.interfaces.vector_database_interface import (
    VectorDatabaseInterface
)

from knowledge.models.chunk import Chunk
from knowledge.models.embedding import Embedding
from knowledge.models.query import Query
from knowledge.models.retrieval_result import RetrievalResult


class ChromaDatabase(VectorDatabaseInterface):

    def __init__(self, path: str | None = None):

        if path is None:
            default_path = Path(__file__).resolve().parent.parent / "chroma_db"
            path = str(default_path)

        self.db_path = str(path)
        self.client = chromadb.PersistentClient(
            path=self.db_path
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
        embedding: Embedding,
        document: str | None = None
    ):

        db = self.create_collection(collection)

        metadata = embedding.metadata

        # ChromaDB requires a non-empty metadata dictionary
        if not metadata:

            metadata = {
                "chunk_id": embedding.chunk_id
            }

        kwargs = {
            "ids": [embedding.chunk_id],
            "embeddings": [embedding.vector],
            "metadatas": [metadata]
        }

        if document is not None:
            kwargs["documents"] = [document]

        db.add(**kwargs)

    def search(
        self,
        query: Query,
        query_embedding: list[float] | None = None
    ) -> RetrievalResult:
        """
        Execute vector similarity search in ChromaDB.

        Score convention:
        - RetrievalResult.scores contains raw Chroma distance metrics (lower = closer match).
        - chunk.metadata["distance"] contains the raw Chroma distance.
        - chunk.metadata["similarity"] contains the normalized similarity score in (0, 1] (higher = closer match).
        - chunk.metadata["collection"] is verified and set to the queried collection name.
        """

        if query.collection:
            collection_name = (
                query.collection.value
                if hasattr(query.collection, "value")
                else str(query.collection)
            )
        else:
            collection_name = "general"

        try:
            col = self.client.get_collection(name=collection_name)
        except Exception:
            return RetrievalResult(
                query=query.text,
                chunks=[],
                scores=[]
            )

        total_count = col.count()
        if total_count == 0 or query.top_k <= 0:
            return RetrievalResult(
                query=query.text,
                chunks=[],
                scores=[]
            )

        n_results = min(query.top_k, total_count)

        if query_embedding is None:
            if not query.text or not query.text.strip():
                return RetrievalResult(
                    query=query.text,
                    chunks=[],
                    scores=[]
                )

            from knowledge.embeddings.sentence_transformer_embedder import (
                SentenceTransformerEmbedder
            )
            embedder = SentenceTransformerEmbedder()
            query_embedding = embedder.embed_query(query.text)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["metadatas", "documents", "distances"]
        }

        if query.filters:
            query_kwargs["where"] = query.filters

        results = col.query(**query_kwargs)

        chunks = []
        scores = []

        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0] if results.get("documents") else []
        metas_list = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        dists_list = results.get("distances", [[]])[0] if results.get("distances") else []

        for i, chunk_id in enumerate(ids_list):
            metadata = dict(metas_list[i]) if i < len(metas_list) and metas_list[i] else {}
            doc_content = docs_list[i] if i < len(docs_list) and docs_list[i] is not None else ""
            if not doc_content and "text" in metadata:
                doc_content = str(metadata["text"])

            distance = float(dists_list[i]) if i < len(dists_list) and dists_list[i] is not None else 0.0
            similarity = 1.0 / (1.0 + distance)

            # Collection metadata consistency:
            # Ensure chunk metadata reflects the collection from which it was queried
            if "collection" in metadata and metadata["collection"] != collection_name:
                metadata["stored_collection"] = metadata["collection"]
            metadata["collection"] = collection_name

            # Explicit distance and similarity metrics
            metadata["distance"] = distance
            metadata["similarity"] = similarity
            metadata["score"] = distance

            chunk = Chunk(
                id=chunk_id,
                document_id=str(metadata.get("document", metadata.get("document_id", ""))),
                text=doc_content,
                chunk_index=int(metadata.get("chunk", metadata.get("chunk_index", 0))),
                metadata=metadata
            )
            chunks.append(chunk)
            scores.append(distance)


        return RetrievalResult(
            query=query.text,
            chunks=chunks,
            scores=scores
        )

    def delete_collection(
        self,
        name: str
    ):

        self.client.delete_collection(name)