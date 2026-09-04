from dataclasses import dataclass, field
from typing import List

from knowledge.models.chunk import Chunk


@dataclass
class KnowledgeContext:
    """
    Structured context built from retrieval results, suitable for LLM consumption.

    Attributes:
        query: Original user query string.
        chunks: Ordered list of deduplicated, non-empty chunks included in the context.
        formatted_context: Deterministic, LLM-ready textual representation of the sources.
    """

    query: str = ""

    chunks: List[Chunk] = field(default_factory=list)

    formatted_context: str = ""

    retrieved_chunks: List[Chunk] = field(default_factory=list)

    prompt: str = ""

    def __post_init__(self):
        # Synchronize chunks and retrieved_chunks for backward compatibility
        if not self.chunks and self.retrieved_chunks:
            self.chunks = self.retrieved_chunks
        elif self.chunks and not self.retrieved_chunks:
            self.retrieved_chunks = self.chunks

        # Synchronize formatted_context and prompt for backward compatibility
        if not self.formatted_context and self.prompt:
            self.formatted_context = self.prompt
        elif self.formatted_context and not self.prompt:
            self.prompt = self.formatted_context


# Model alias consistent with project architecture
Context = KnowledgeContext