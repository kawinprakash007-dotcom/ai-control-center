from abc import ABC, abstractmethod

from knowledge.models.chunk import Chunk
from knowledge.models.embedding import Embedding


class EmbedderInterface(ABC):

    @abstractmethod
    def embed(self, chunk: Chunk) -> Embedding:
        pass

    @abstractmethod
    def embed_batch(
        self,
        chunks: list[Chunk]
    ) -> list[Embedding]:
        pass