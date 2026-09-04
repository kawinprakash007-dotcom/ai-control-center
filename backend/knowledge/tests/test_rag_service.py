from typing import Optional

from knowledge.interfaces.retriever_interface import RetrieverInterface
from knowledge.interfaces.context_builder_interface import ContextBuilderInterface
from knowledge.models.query import Query
from knowledge.models.chunk import Chunk
from knowledge.models.retrieval_result import RetrievalResult
from knowledge.models.knowledge_context import KnowledgeContext
from knowledge.services.rag_service import RAGService


def test_rag_service():
    print("=" * 60)
    print("RUNNING RAG SERVICE TESTS")
    print("=" * 60)

    # -------------------------------------------------------------
    # Test 1: Real Linux query
    # -------------------------------------------------------------
    print("\n[Test 1] Real Linux query: 'Linux process'")
    real_rag = RAGService()
    context = real_rag.query("Linux process")

    assert isinstance(context, KnowledgeContext), "Context must be KnowledgeContext"
    assert context.query == "Linux process", f"Expected query 'Linux process', got '{context.query}'"
    assert len(context.chunks) > 0, "Must retrieve at least 1 chunk from real Linux collection"
    assert "Linux.pdf" in context.formatted_context, "Expected 'Linux.pdf' in formatted_context"
    assert len(context.formatted_context.strip()) > 0, "formatted_context must be non-empty"
    print("  [OK] Test 1 PASSED: Real Linux query successfully orchestrated!")

    # -------------------------------------------------------------
    # Test 2: Dependency injection using mocked/fake components
    # -------------------------------------------------------------
    print("\n[Test 2] Dependency injection with mocked/fake components")
    fake_chunk = Chunk(
        id="fake-chunk-1",
        document_id="Guide.pdf",
        text="Sample fake content",
        chunk_index=0,
        metadata={"similarity": 0.95}
    )
    fake_retrieval_result = RetrievalResult(
        query="di query",
        chunks=[fake_chunk],
        scores=[0.05]
    )

    class FakeRetriever(RetrieverInterface):
        def __init__(self):
            self.call_count = 0
            self.last_query = None

        def retrieve(self, query: Query) -> RetrievalResult:
            self.call_count += 1
            self.last_query = query
            return fake_retrieval_result

    class FakeContextBuilder(ContextBuilderInterface):
        def __init__(self):
            self.received_result = None
            self.stub_context = KnowledgeContext(
                query="di query",
                chunks=[fake_chunk],
                formatted_context="[Source 1]\nDocument: Guide.pdf\nContent:\nSample fake content"
            )

        def build(self, result: Optional[RetrievalResult]) -> KnowledgeContext:
            self.received_result = result
            return self.stub_context

    fake_retriever = FakeRetriever()
    fake_builder = FakeContextBuilder()
    di_rag = RAGService(retriever=fake_retriever, context_builder=fake_builder)

    test_query = Query(text="di query", collection="test")
    res_context = di_rag.query(test_query)

    assert fake_retriever.call_count == 1, f"Expected 1 retrieval call, got {fake_retriever.call_count}"
    assert fake_builder.received_result is fake_retrieval_result, "Context builder did not receive exact RetrievalResult"
    assert res_context is fake_builder.stub_context, "Returned KnowledgeContext must match builder output"
    print("  [OK] Test 2 PASSED: Dependency injection orchestration verified!")

    # -------------------------------------------------------------
    # Test 3: Empty query (graceful handling without ChromaDB access)
    # -------------------------------------------------------------
    print("\n[Test 3] Empty query graceful handling without ChromaDB access")
    class GuardRetriever(RetrieverInterface):
        def retrieve(self, query: Query) -> RetrievalResult:
            raise AssertionError("Retriever must NOT be called for empty queries!")

    guard_rag = RAGService(retriever=GuardRetriever())
    for empty_input in ["", "   ", Query(text=""), Query(text="   \n\t "), None]:
        ctx = guard_rag.query(empty_input)
        assert isinstance(ctx, KnowledgeContext), "Must return KnowledgeContext"
        assert len(ctx.chunks) == 0, f"Expected 0 chunks for {empty_input!r}, got {len(ctx.chunks)}"
        assert ctx.formatted_context == "", f"Expected empty formatted_context for {empty_input!r}"
    print("  [OK] Test 3 PASSED: Empty queries handled gracefully without invoking retriever!")

    # -------------------------------------------------------------
    # Test 4: Query with no retrieval results
    # -------------------------------------------------------------
    print("\n[Test 4] Query with no retrieval results")
    class EmptyRetriever(RetrieverInterface):
        def retrieve(self, query: Query) -> RetrievalResult:
            return RetrievalResult(query=query.text, chunks=[], scores=[])

    empty_rag = RAGService(retriever=EmptyRetriever())
    ctx_empty = empty_rag.query("query with zero results")

    assert isinstance(ctx_empty, KnowledgeContext)
    assert len(ctx_empty.chunks) == 0
    assert ctx_empty.formatted_context == ""
    assert ctx_empty.query == "query with zero results"
    print("  [OK] Test 4 PASSED: Empty retrieval results return empty context safely!")

    # -------------------------------------------------------------
    # Test 5: Retrieval failure handled without fabricating context
    # -------------------------------------------------------------
    print("\n[Test 5] Retrieval failure handled without fabricating context")
    class FailingRetriever(RetrieverInterface):
        def retrieve(self, query: Query) -> RetrievalResult:
            raise ConnectionError("ChromaDB service connection dropped")

    failing_rag = RAGService(retriever=FailingRetriever())
    ctx_fail = failing_rag.query(Query(text="failing query", collection="linux"))

    assert isinstance(ctx_fail, KnowledgeContext)
    assert len(ctx_fail.chunks) == 0, "Must not fabricate chunks on retrieval failure"
    assert ctx_fail.formatted_context == "", "Must not fabricate text on retrieval failure"
    assert ctx_fail.query == "failing query"
    print("  [OK] Test 5 PASSED: Retrieval failure handled cleanly without fabricating context!")

    # -------------------------------------------------------------
    # Test 6: Context builder failure handled cleanly
    # -------------------------------------------------------------
    print("\n[Test 6] Context builder failure handled cleanly")
    class FailingBuilder(ContextBuilderInterface):
        def build(self, result: Optional[RetrievalResult]) -> KnowledgeContext:
            raise ValueError("Context formatting overflow error")

    builder_fail_rag = RAGService(
        retriever=FakeRetriever(),
        context_builder=FailingBuilder()
    )
    ctx_builder_fail = builder_fail_rag.query("builder failure query")

    assert isinstance(ctx_builder_fail, KnowledgeContext)
    assert len(ctx_builder_fail.chunks) == 0
    assert ctx_builder_fail.formatted_context == ""
    assert ctx_builder_fail.query == "builder failure query"
    print("  [OK] Test 6 PASSED: Context builder failure handled cleanly!")

    # -------------------------------------------------------------
    # Test 7: Original Query remains unchanged
    # -------------------------------------------------------------
    print("\n[Test 7] Original Query remains unchanged")
    original_query = Query(
        text="Immutable Linux Query",
        collection="linux",
        top_k=4,
        filters={"type": "pdf"}
    )
    orig_text = original_query.text
    orig_col = original_query.collection
    orig_topk = original_query.top_k
    orig_filters = dict(original_query.filters)

    audit_rag_instance = RAGService()
    audit_rag_instance.query(original_query)

    assert original_query.text == orig_text, "Query text was mutated!"
    assert original_query.collection == orig_col, "Query collection was mutated!"
    assert original_query.top_k == orig_topk, "Query top_k was mutated!"
    assert original_query.filters == orig_filters, "Query filters were mutated!"
    print("  [OK] Test 7 PASSED: Original Query object remained completely immutable!")

    # -------------------------------------------------------------
    # Test 8: RAGService does not modify RetrievalResult or KnowledgeContext
    # -------------------------------------------------------------
    print("\n[Test 8] Inputs/outputs not mutated unexpectedly")
    sample_chunk = Chunk(
        id="chunk-audit-1",
        document_id="Audit.txt",
        text="Audit test text",
        chunk_index=0,
        metadata={"tag": "v1"}
    )
    sample_result = RetrievalResult(
        query="audit query",
        chunks=[sample_chunk],
        scores=[0.42]
    )

    chunks_len_before = len(sample_result.chunks)
    scores_len_before = len(sample_result.scores)
    metadata_before = dict(sample_chunk.metadata)

    class AuditRetriever(RetrieverInterface):
        def retrieve(self, query: Query) -> RetrievalResult:
            return sample_result

    class AuditBuilder(ContextBuilderInterface):
        def build(self, result: Optional[RetrievalResult]) -> KnowledgeContext:
            assert len(result.chunks) == chunks_len_before, "RetrievalResult chunks mutated before reaching builder!"
            assert len(result.scores) == scores_len_before, "RetrievalResult scores mutated before reaching builder!"
            assert result.chunks[0].metadata == metadata_before, "Chunk metadata mutated before reaching builder!"
            return KnowledgeContext(query=result.query, chunks=result.chunks, formatted_context="audit output")

    audit_rag = RAGService(retriever=AuditRetriever(), context_builder=AuditBuilder())
    audit_context = audit_rag.query("audit query")

    assert len(sample_result.chunks) == chunks_len_before, "sample_result.chunks length was mutated!"
    assert len(sample_result.scores) == scores_len_before, "sample_result.scores length was mutated!"
    assert sample_result.chunks[0].metadata == metadata_before, "sample_chunk metadata was mutated!"
    assert audit_context.formatted_context == "audit output"
    print("  [OK] Test 8 PASSED: Components and results were not modified unexpectedly!")

    print("\n" + "=" * 60)
    print("ALL 8 RAG SERVICE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_rag_service()
