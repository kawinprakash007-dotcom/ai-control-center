from abc import ABC, abstractmethod

from knowledge.models.document import Document
from knowledge.models.chunk import Chunk


class Chunker(ABC):
    """
    Base interface for all chunking strategies.
    """

    @abstractmethod
    def chunk(
        self,
        document: Document
    ) -> list[Chunk]:
        """
        Split a document into chunks.
        """
        pass