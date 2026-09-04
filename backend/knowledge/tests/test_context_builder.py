import copy
from knowledge.models.query import Query
from knowledge.models.chunk import Chunk
from knowledge.models.retrieval_result import RetrievalResult
from knowledge.models.knowledge_context import KnowledgeContext
from knowledge.retrieval.retriever import KnowledgeRetriever
from knowledge.context.context_builder import ContextBuilder


def test_context_builder():
    print("=" * 60)
    print("RUNNING CONTEXT BUILDER TESTS")
    print("=" * 60)

    # -------------------------------------------------------------
    # Test 1: Build context from the actual Linux retrieval result
    # -------------------------------------------------------------
    print("\n[Test 1] Build context from actual Linux retrieval result")
    retriever = KnowledgeRetriever()
    query = Query(text="Linux process", collection="linux", top_k=1)
    retrieval_result = retriever.retrieve(query)

    assert len(retrieval_result.chunks) > 0, "Prerequisite: Linux collection must return at least 1 chunk"

    builder = ContextBuilder()
    context = builder.build(retrieval_result)

    assert isinstance(context, KnowledgeContext), "Context must be an instance of KnowledgeContext"
    assert context.query == "Linux process", f"Expected query 'Linux process', got '{context.query}'"
    assert "Linux.pdf" in context.formatted_context, "Expected 'Linux.pdf' to appear in formatted_context"
    assert retrieval_result.chunks[0].text in context.formatted_context, "Expected chunk text to appear in formatted_context"
    assert len(context.formatted_context.strip()) > 0, "formatted_context must be non-empty"
    assert len(context.chunks) == 1, f"Expected 1 chunk, got {len(context.chunks)}"
    assert "[Source 1]" in context.formatted_context, "Expected '[Source 1]' in formatted_context"
    assert "Relevance:" in context.formatted_context, "Expected 'Relevance:' in formatted_context"
    print("  [OK] Test 1 PASSED: Context successfully built from real Linux retrieval result!")

    # -------------------------------------------------------------
    # Test 2: Multiple chunks preserve retrieval order
    # -------------------------------------------------------------
    print("\n[Test 2] Multiple chunks preserve retrieval order")
    c1 = Chunk(id="chunk-alpha", document_id="DocA.txt", text="First chunk content", chunk_index=0, metadata={"score": 0.1})
    c2 = Chunk(id="chunk-beta", document_id="DocB.txt", text="Second chunk content", chunk_index=1, metadata={"score": 0.2})
    c3 = Chunk(id="chunk-gamma", document_id="DocC.txt", text="Third chunk content", chunk_index=2, metadata={"score": 0.3})

    result_multi = RetrievalResult(
        query="test ordering",
        chunks=[c1, c2, c3],
        scores=[0.1, 0.2, 0.3]
    )

    context_ordered = ContextBuilder().build(result_multi)
    assert len(context_ordered.chunks) == 3, f"Expected 3 chunks, got {len(context_ordered.chunks)}"
    assert context_ordered.chunks[0].id == "chunk-alpha"
    assert context_ordered.chunks[1].id == "chunk-beta"
    assert context_ordered.chunks[2].id == "chunk-gamma"

    # Verify order in formatted_context string
    pos1 = context_ordered.formatted_context.find("First chunk content")
    pos2 = context_ordered.formatted_context.find("Second chunk content")
    pos3 = context_ordered.formatted_context.find("Third chunk content")
    assert 0 <= pos1 < pos2 < pos3, "Chunks must appear in formatted_context in strictly sequential order"
    print("  [OK] Test 2 PASSED: Multiple chunks strictly preserve retrieval order!")

    # -------------------------------------------------------------
    # Test 3: max_chunks limits output
    # -------------------------------------------------------------
    print("\n[Test 3] max_chunks limits output")
    builder_max2 = ContextBuilder(max_chunks=2)
    context_max2 = builder_max2.build(result_multi)

    assert len(context_max2.chunks) == 2, f"Expected 2 chunks, got {len(context_max2.chunks)}"
    assert context_max2.chunks[0].id == "chunk-alpha"
    assert context_max2.chunks[1].id == "chunk-beta"
    assert "First chunk content" in context_max2.formatted_context
    assert "Second chunk content" in context_max2.formatted_context
    assert "Third chunk content" not in context_max2.formatted_context
    assert "[Source 1]" in context_max2.formatted_context
    assert "[Source 2]" in context_max2.formatted_context
    assert "[Source 3]" not in context_max2.formatted_context
    print("  [OK] Test 3 PASSED: max_chunks=2 accurately limits chunks and formatted output!")

    # -------------------------------------------------------------
    # Test 4: max_characters limits output
    # -------------------------------------------------------------
    print("\n[Test 4] max_characters limits output")
    # Build with unlimited to measure single and double chunk sizes
    unlimited_fmt = ContextBuilder().build(result_multi).formatted_context
    # Set limit that only allows the first chunk
    first_chunk_only_fmt = ContextBuilder(max_chunks=1).build(result_multi).formatted_context
    tight_char_limit = len(first_chunk_only_fmt) + 10

    builder_char_limited = ContextBuilder(max_characters=tight_char_limit)
    context_char_limited = builder_char_limited.build(result_multi)

    assert len(context_char_limited.chunks) == 1, f"Expected 1 chunk under character limit, got {len(context_char_limited.chunks)}"
    assert len(context_char_limited.formatted_context) <= tight_char_limit, (
        f"Formatted context length {len(context_char_limited.formatted_context)} exceeds limit {tight_char_limit}"
    )
    assert context_char_limited.chunks[0].id == "chunk-alpha"
    assert "Second chunk content" not in context_char_limited.formatted_context
    print(f"  [OK] Test 4 PASSED: max_characters={tight_char_limit} enforced (context length: {len(context_char_limited.formatted_context)})!")

    # -------------------------------------------------------------
    # Test 5: Duplicate chunk IDs are removed
    # -------------------------------------------------------------
    print("\n[Test 5] Duplicate chunk IDs are removed")
    c_dup1 = Chunk(id="chunk-1", document_id="Doc.txt", text="Content 1", chunk_index=0)
    c_dup2 = Chunk(id="chunk-2", document_id="Doc.txt", text="Content 2", chunk_index=1)
    c_dup1_repeat = Chunk(id="chunk-1", document_id="Doc.txt", text="Content 1 duplicate", chunk_index=0)
    c_dup3 = Chunk(id="chunk-3", document_id="Doc.txt", text="Content 3", chunk_index=2)
    c_dup2_repeat = Chunk(id="chunk-2", document_id="Doc.txt", text="Content 2 duplicate", chunk_index=1)

    result_duplicates = RetrievalResult(
        query="test deduplication",
        chunks=[c_dup1, c_dup2, c_dup1_repeat, c_dup3, c_dup2_repeat],
        scores=[0.1, 0.2, 0.3, 0.4, 0.5]
    )

    context_dedup = ContextBuilder().build(result_duplicates)
    dedup_ids = [c.id for c in context_dedup.chunks]
    assert dedup_ids == ["chunk-1", "chunk-2", "chunk-3"], f"Expected ['chunk-1', 'chunk-2', 'chunk-3'], got {dedup_ids}"
    assert len(context_dedup.chunks) == 3
    assert "[Source 1]" in context_dedup.formatted_context
    assert "[Source 2]" in context_dedup.formatted_context
    assert "[Source 3]" in context_dedup.formatted_context
    assert "[Source 4]" not in context_dedup.formatted_context
    print("  [OK] Test 5 PASSED: Duplicate chunks successfully removed while preserving first occurrence!")

    # -------------------------------------------------------------
    # Test 6: Empty RetrievalResult returns empty context safely
    # -------------------------------------------------------------
    print("\n[Test 6] Empty RetrievalResult returns empty context safely")
    empty_result = RetrievalResult(query="empty query", chunks=[], scores=[])
    context_empty = ContextBuilder().build(empty_result)

    assert context_empty.query == "empty query"
    assert len(context_empty.chunks) == 0
    assert context_empty.formatted_context == ""

    # Also test None RetrievalResult
    context_none = ContextBuilder().build(None)
    assert context_none.query == ""
    assert len(context_none.chunks) == 0
    assert context_none.formatted_context == ""
    print("  [OK] Test 6 PASSED: Empty and None RetrievalResults handled gracefully!")

    # -------------------------------------------------------------
    # Test 7: Chunk with empty text is skipped safely
    # -------------------------------------------------------------
    print("\n[Test 7] Chunk with empty text is skipped safely")
    c_valid1 = Chunk(id="v1", document_id="Doc.txt", text="Valid text 1", chunk_index=0)
    c_empty1 = Chunk(id="e1", document_id="Doc.txt", text="", chunk_index=1)
    c_whitespace = Chunk(id="e2", document_id="Doc.txt", text="   \n\t  \n  ", chunk_index=2)
    c_valid2 = Chunk(id="v2", document_id="Doc.txt", text="Valid text 2", chunk_index=3)

    result_with_empty = RetrievalResult(
        query="test empty text",
        chunks=[c_valid1, c_empty1, c_whitespace, c_valid2],
        scores=[0.1, 0.2, 0.3, 0.4]
    )

    context_filtered = ContextBuilder().build(result_with_empty)
    assert len(context_filtered.chunks) == 2, f"Expected 2 non-empty chunks, got {len(context_filtered.chunks)}"
    assert context_filtered.chunks[0].id == "v1"
    assert context_filtered.chunks[1].id == "v2"
    assert "[Source 1]" in context_filtered.formatted_context
    assert "[Source 2]" in context_filtered.formatted_context
    assert "[Source 3]" not in context_filtered.formatted_context
    print("  [OK] Test 7 PASSED: Empty and whitespace-only chunks safely omitted!")

    # -------------------------------------------------------------
    # Test 8: Missing metadata/score/document fields do not crash
    # -------------------------------------------------------------
    print("\n[Test 8] Missing metadata/score/document fields do not crash")
    c_missing = Chunk(
        id="c-missing",
        document_id="",
        text="Minimal chunk text without document or metadata",
        chunk_index=0,
        metadata={}
    )

    result_missing = RetrievalResult(
        query="missing fields query",
        chunks=[c_missing],
        scores=[]  # No scores provided
    )

    context_missing = ContextBuilder().build(result_missing)
    assert len(context_missing.chunks) == 1
    assert "Minimal chunk text without document or metadata" in context_missing.formatted_context
    assert "Document: Unknown" in context_missing.formatted_context
    assert "Relevance: N/A" in context_missing.formatted_context
    print("  [OK] Test 8 PASSED: Missing metadata, score, and document fields handled without error!")

    # -------------------------------------------------------------
    # Test 9: Invalid limits (0 and negative values) are handled safely
    # -------------------------------------------------------------
    print("\n[Test 9] Invalid limits (0 and negative values) are handled safely")
    builder_zero_chunks = ContextBuilder(max_chunks=0)
    builder_neg_chunks = ContextBuilder(max_chunks=-3)
    builder_zero_chars = ContextBuilder(max_characters=0)
    builder_neg_chars = ContextBuilder(max_characters=-100)

    for b in [builder_zero_chunks, builder_neg_chunks, builder_zero_chars, builder_neg_chars]:
        ctx = b.build(result_multi)
        assert len(ctx.chunks) == 0, f"Expected 0 chunks for invalid limit, got {len(ctx.chunks)}"
        assert ctx.formatted_context == "", f"Expected empty formatted_context for invalid limit, got '{ctx.formatted_context}'"
        assert ctx.query == "test ordering", "Query must still be preserved"
    print("  [OK] Test 9 PASSED: Zero and negative limits handled safely with empty context!")

    # -------------------------------------------------------------
    # Test 10: ContextBuilder does not modify the original RetrievalResult
    # -------------------------------------------------------------
    print("\n[Test 10] ContextBuilder does not modify the original RetrievalResult")
    orig_chunk1 = Chunk(id="orig-1", document_id="Doc.txt", text="Original text 1", chunk_index=0, metadata={"tag": "initial"})
    orig_chunk2 = Chunk(id="orig-2", document_id="Doc.txt", text="Original text 2", chunk_index=1, metadata={"tag": "initial"})
    original_result = RetrievalResult(
        query="original query",
        chunks=[orig_chunk1, orig_chunk2],
        scores=[0.15, 0.85]
    )

    # Make deep copy snapshot of chunks and scores
    chunks_before_count = len(original_result.chunks)
    scores_before_count = len(original_result.scores)
    chunk1_meta_before = dict(orig_chunk1.metadata)
    chunk2_meta_before = dict(orig_chunk2.metadata)

    # Build context with limits and scores
    builder_audit = ContextBuilder(max_chunks=1)
    context_audit = builder_audit.build(original_result)

    # Verify original_result wasn't touched
    assert len(original_result.chunks) == chunks_before_count, "original_result.chunks length was modified!"
    assert len(original_result.scores) == scores_before_count, "original_result.scores length was modified!"
    assert original_result.chunks[0].metadata == chunk1_meta_before, "original_result chunk 0 metadata was mutated!"
    assert original_result.chunks[1].metadata == chunk2_meta_before, "original_result chunk 1 metadata was mutated!"
    assert "score" not in original_result.chunks[0].metadata, "original chunk metadata was polluted with 'score'!"

    # Verify context chunk has the score properly attached without altering original
    assert "score" in context_audit.chunks[0].metadata
    print("  [OK] Test 10 PASSED: ContextBuilder guarantees complete immutability of RetrievalResult!")

    print("\n" + "=" * 60)
    print("ALL 10 CONTEXT BUILDER TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_context_builder()
