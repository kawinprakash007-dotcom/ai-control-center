from dataclasses import dataclass
from typing import List

from knowledge.models.chunk import Chunk


@dataclass
class RetrievalResult:
    """
    Result of a vector similarity retrieval operation.

    Attributes:
        query: The input query text.
        chunks: List of retrieved Chunk instances with metadata and identifiers.
        scores: List of raw Chroma distance metrics corresponding to each chunk
                (e.g. L2/Euclidean distance where lower values indicate closer matches).
                Normalized similarity scores (1 / (1 + distance), where higher values
                in (0, 1] indicate closer matches) are accessible via chunk.metadata["similarity"].
    """

    query: str

    chunks: List[Chunk]

    scores: List[float]