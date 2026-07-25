from knowledge.interfaces.embedder_interface import EmbedderInterface

from knowledge.models.chunk import Chunk
from knowledge.models.embedding import Embedding


class EmbeddingService:

    def __init__(self, embedder: EmbedderInterface):

        self.embedder = embedder

    def embed_chunk(
        self,
        chunk: Chunk
    ) -> Embedding:

        return self.embedder.embed(chunk)

    def embed_chunks(
        self,
        chunks: list[Chunk]
    ) -> list[Embedding]:

        return self.embedder.embed_batch(chunks)