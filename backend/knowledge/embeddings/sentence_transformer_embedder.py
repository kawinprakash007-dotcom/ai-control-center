from knowledge.interfaces.embedder_interface import EmbedderInterface

from knowledge.models.chunk import Chunk
from knowledge.models.embedding import Embedding

from knowledge.embeddings.embedding_model import EmbeddingModel


class SentenceTransformerEmbedder(EmbedderInterface):

    def __init__(self):

        self.model = EmbeddingModel()

    def embed(

        self,

        chunk: Chunk

    ) -> Embedding:

        vector = self.model.encode(

            chunk.text

        )

        return Embedding(

            chunk_id=chunk.id,

            vector=vector.tolist(),

            metadata=chunk.metadata

        )

    def embed_batch(

        self,

        chunks: list[Chunk]

    ) -> list[Embedding]:

        embeddings = []

        for chunk in chunks:

            embeddings.append(

                self.embed(chunk)

            )

        return embeddings

    def embed_query(
        self,
        text: str
    ) -> list[float]:

        vector = self.model.encode(text)

        return vector.tolist()