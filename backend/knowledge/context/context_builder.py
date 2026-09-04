from typing import Optional, List, Set, Any

from knowledge.interfaces.context_builder_interface import ContextBuilderInterface
from knowledge.models.chunk import Chunk
from knowledge.models.retrieval_result import RetrievalResult
from knowledge.models.knowledge_context import KnowledgeContext


class ContextBuilder(ContextBuilderInterface):
    """
    Context Builder for AI Control Center Knowledge Engine.

    Transforms raw RetrievalResult instances into a clean, structured,
    deterministic KnowledgeContext suitable for LLM consumption.

    Responsibilities:
    - Preserves query and chunk ranking order
    - Deduplicates chunks by chunk ID (with fallback)
    - Filters out empty or whitespace-only chunks
    - Formats sources deterministically with metadata and relevance scores
    - Enforces max_chunks and max_characters budget constraints
    - Guarantees immutability of the original RetrievalResult
    """

    def __init__(
        self,
        max_chunks: Optional[int] = None,
        max_characters: Optional[int] = None
    ):
        """
        Initialize ContextBuilder with optional budget constraints.

        Args:
            max_chunks: Maximum number of chunks to include in context (None for unlimited).
            max_characters: Maximum total character count for formatted_context (None for unlimited).
        """
        self.max_chunks = max_chunks
        self.max_characters = max_characters

    def build(
        self,
        retrieval_result: Optional[RetrievalResult]
    ) -> KnowledgeContext:
        """
        Transform a RetrievalResult into a structured KnowledgeContext.

        Args:
            retrieval_result: The raw retrieval result from KnowledgeRetriever.

        Returns:
            KnowledgeContext containing preserved query, selected chunks, and formatted text.
        """
        if retrieval_result is None:
            return KnowledgeContext(query="", chunks=[], formatted_context="")

        query_text = retrieval_result.query if retrieval_result.query is not None else ""

        # Handle invalid budget constraints safely (0 or negative)
        if self.max_chunks is not None and self.max_chunks <= 0:
            return KnowledgeContext(query=query_text, chunks=[], formatted_context="")

        if self.max_characters is not None and self.max_characters <= 0:
            return KnowledgeContext(query=query_text, chunks=[], formatted_context="")

        if not retrieval_result.chunks:
            return KnowledgeContext(query=query_text, chunks=[], formatted_context="")

        selected_chunks: List[Chunk] = []
        formatted_chunks: List[str] = []
        seen_ids: Set[str] = set()

        scores = retrieval_result.scores if retrieval_result.scores else []

        for i, original_chunk in enumerate(retrieval_result.chunks):
            if original_chunk is None:
                continue

            # Safely skip empty or whitespace-only text chunks
            if not original_chunk.text or not str(original_chunk.text).strip():
                continue

            # Deduplication key: primary is chunk.id, with deterministic fallback
            chunk_id = str(original_chunk.id).strip() if original_chunk.id else ""
            if not chunk_id:
                doc_id = original_chunk.document_id or ""
                idx = original_chunk.chunk_index if original_chunk.chunk_index is not None else ""
                chunk_id = f"{doc_id}:{idx}:{hash(original_chunk.text)}"

            if chunk_id in seen_ids:
                continue

            # Enforce max_chunks constraint
            if self.max_chunks is not None and len(selected_chunks) >= self.max_chunks:
                break

            score = scores[i] if i < len(scores) else None

            # Preserve immutability: deep-copy metadata and construct new context chunk
            chunk_metadata = dict(original_chunk.metadata) if original_chunk.metadata else {}
            if score is not None:
                if "score" not in chunk_metadata:
                    chunk_metadata["score"] = score
                if "distance" not in chunk_metadata:
                    chunk_metadata["distance"] = score

            context_chunk = Chunk(
                id=original_chunk.id,
                document_id=original_chunk.document_id,
                text=original_chunk.text,
                chunk_index=original_chunk.chunk_index,
                metadata=chunk_metadata
            )

            # Format candidate chunk representation
            chunk_str = self._format_chunk(
                index=len(selected_chunks) + 1,
                chunk=context_chunk,
                score=score
            )

            # Enforce max_characters constraint
            candidate_formatted = (
                "\n\n".join(formatted_chunks + [chunk_str])
                if formatted_chunks
                else chunk_str
            )

            if self.max_characters is not None and len(candidate_formatted) > self.max_characters:
                # Stop adding chunks when character limit is reached
                break

            seen_ids.add(chunk_id)
            selected_chunks.append(context_chunk)
            formatted_chunks.append(chunk_str)

        formatted_context = "\n\n".join(formatted_chunks) if formatted_chunks else ""

        return KnowledgeContext(
            query=query_text,
            chunks=selected_chunks,
            formatted_context=formatted_context
        )

    def build_context(
        self,
        retrieval_result: Optional[RetrievalResult]
    ) -> KnowledgeContext:
        """Alias for build()."""
        return self.build(retrieval_result)

    def _resolve_document_name(self, chunk: Chunk) -> str:
        """Extract a readable document name from chunk or its metadata."""
        if chunk.document_id and str(chunk.document_id).strip():
            return str(chunk.document_id).strip()
        if chunk.metadata:
            if "document" in chunk.metadata and str(chunk.metadata["document"]).strip():
                return str(chunk.metadata["document"]).strip()
            if "document_id" in chunk.metadata and str(chunk.metadata["document_id"]).strip():
                return str(chunk.metadata["document_id"]).strip()
            if "source" in chunk.metadata and str(chunk.metadata["source"]).strip():
                return str(chunk.metadata["source"]).strip()
        return "Unknown"

    def _resolve_chunk_index(self, chunk: Chunk) -> int:
        """Extract chunk index safely."""
        if chunk.chunk_index is not None:
            try:
                return int(chunk.chunk_index)
            except (ValueError, TypeError):
                pass
        if chunk.metadata:
            for key in ("chunk", "chunk_index", "index"):
                if key in chunk.metadata and chunk.metadata[key] is not None:
                    try:
                        return int(chunk.metadata[key])
                    except (ValueError, TypeError):
                        pass
        return 0

    def _resolve_relevance_string(
        self,
        chunk: Chunk,
        score: Optional[float] = None
    ) -> str:
        """Format relevance/similarity score deterministically."""
        val = None
        if chunk.metadata:
            if "similarity" in chunk.metadata and chunk.metadata["similarity"] is not None:
                val = chunk.metadata["similarity"]
            elif "score" in chunk.metadata and chunk.metadata["score"] is not None:
                val = chunk.metadata["score"]
            elif "distance" in chunk.metadata and chunk.metadata["distance"] is not None:
                val = chunk.metadata["distance"]

        if val is None and score is not None:
            val = score

        if val is not None:
            try:
                return f"{float(val):.4f}"
            except (ValueError, TypeError):
                return str(val)
        return "N/A"

    def _format_chunk(
        self,
        index: int,
        chunk: Chunk,
        score: Optional[float] = None
    ) -> str:
        """Format a single source chunk deterministically."""
        doc_name = self._resolve_document_name(chunk)
        chunk_idx = self._resolve_chunk_index(chunk)
        relevance_str = self._resolve_relevance_string(chunk, score)
        content_text = chunk.text.strip() if chunk.text else ""

        return (
            f"[Source {index}]\n"
            f"Document: {doc_name}\n"
            f"Chunk: {chunk_idx}\n"
            f"Relevance: {relevance_str}\n"
            f"Content:\n"
            f"{content_text}"
        )
