from knowledge.models.query import Query
from knowledge.models.retrieval_result import RetrievalResult
from knowledge.retrieval.retriever import KnowledgeRetriever
from knowledge.services.retrieval_service import RetrievalService


def test_knowledge_retriever():
    retriever = KnowledgeRetriever()
    print("=" * 60)
    print("RUNNING KNOWLEDGE RETRIEVER TESTS")
    print("=" * 60)

    # -------------------------------------------------------------
    # Test 1: Normal query against indexed Linux.pdf collection
    # -------------------------------------------------------------
    print("\n[Test 1] Normal Query: 'Linux process' on collection 'linux'")
    query = Query(
        text="Linux process",
        collection="linux",
        top_k=1
    )
    result = retriever.retrieve(query)

    assert isinstance(result, RetrievalResult), "Result must be a RetrievalResult instance"
    assert result.query == "Linux process", f"Expected query 'Linux process', got '{result.query}'"
    assert len(result.chunks) == 1, f"Expected 1 chunk retrieved, got {len(result.chunks)}"
    assert len(result.scores) == 1, f"Expected 1 score retrieved, got {len(result.scores)}"

    chunk = result.chunks[0]
    distance = result.scores[0]
    similarity = chunk.metadata.get("similarity")

    print(f"  [OK] Retrieved Chunk ID    : {chunk.id}")
    print(f"  [OK] Document ID           : {chunk.document_id}")
    print(f"  [OK] Chunk Index           : {chunk.chunk_index}")
    print(f"  [OK] Chroma Distance       : {distance}")
    print(f"  [OK] Normalized Similarity : {similarity}")
    print(f"  [OK] Metadata              : {chunk.metadata}")

    # Assert document & collection metadata consistency
    assert chunk.metadata.get("document") == "Linux.pdf", "Expected document name 'Linux.pdf' in metadata"
    assert chunk.metadata.get("collection") == "linux", f"Expected collection 'linux', got '{chunk.metadata.get('collection')}'"

    # Explicit assertions for Chroma distance
    assert isinstance(distance, float), "Chroma distance must be a float"
    assert distance >= 0.0, "Chroma distance must be non-negative"
    assert chunk.metadata.get("distance") == distance, "chunk.metadata['distance'] must match result.scores"

    # Explicit assertions for similarity
    assert isinstance(similarity, float), "Similarity score must be a float"
    assert 0.0 < similarity <= 1.0, "Normalized similarity score must be in (0.0, 1.0]"
    expected_sim = 1.0 / (1.0 + distance)
    assert abs(similarity - expected_sim) < 1e-6, f"Similarity {similarity} must equal 1 / (1 + distance) {expected_sim}"

    # Score field consistency
    assert chunk.metadata.get("score") == distance, "chunk.metadata['score'] must match result.scores"

    print("  [OK] Test 1 PASSED: Successfully retrieved chunk with verified collection, distance, and similarity!")


    # -------------------------------------------------------------
    # Test 2: Configurable top_k
    # -------------------------------------------------------------
    print("\n[Test 2] Configurable top_k (top_k=3 on 1-item collection)")
    query_topk = Query(
        text="Linux kernel",
        collection="linux",
        top_k=3
    )
    result_topk = retriever.retrieve(query_topk)
    # Since the collection currently has 1 item, it should clamp to 1 without error
    assert len(result_topk.chunks) == 1, f"Expected 1 chunk due to collection count clamping, got {len(result_topk.chunks)}"
    print(f"  [OK] Clamped top_k returned {len(result_topk.chunks)} chunk(s) safely.")
    print("  [OK] Test 2 PASSED")

    # -------------------------------------------------------------
    # Test 3: Edge Case - Empty Query
    # -------------------------------------------------------------
    print("\n[Test 3] Empty Query Edge Cases ('', '   ')")
    empty_result_1 = retriever.retrieve(Query(text="", collection="linux"))
    assert len(empty_result_1.chunks) == 0, "Empty query must return 0 chunks"
    assert len(empty_result_1.scores) == 0, "Empty query must return 0 scores"

    empty_result_2 = retriever.retrieve(Query(text="   ", collection="linux"))
    assert len(empty_result_2.chunks) == 0, "Whitespace query must return 0 chunks"
    print("  [OK] Empty query gracefully returned 0 chunks without error.")
    print("  [OK] Test 3 PASSED")

    # -------------------------------------------------------------
    # Test 4: Edge Case - Missing / Invalid Collection
    # -------------------------------------------------------------
    print("\n[Test 4] Missing/Invalid Collection ('non_existent_collection_xyz')")
    missing_col_result = retriever.retrieve(Query(
        text="Linux process",
        collection="non_existent_collection_xyz"
    ))
    assert len(missing_col_result.chunks) == 0, "Missing collection must return 0 chunks"
    assert len(missing_col_result.scores) == 0, "Missing collection must return 0 scores"
    print("  [OK] Missing collection handled gracefully without crash.")
    print("  [OK] Test 4 PASSED")

    # -------------------------------------------------------------
    # Test 5: Edge Case - Invalid top_k (0 or negative)
    # -------------------------------------------------------------
    print("\n[Test 5] Invalid top_k (0 and -5)")
    invalid_topk_1 = retriever.retrieve(Query(text="Linux", collection="linux", top_k=0))
    assert len(invalid_topk_1.chunks) == 0, "top_k=0 must return 0 chunks"

    invalid_topk_2 = retriever.retrieve(Query(text="Linux", collection="linux", top_k=-5))
    assert len(invalid_topk_2.chunks) == 0, "Negative top_k must return 0 chunks"
    print("  [OK] Invalid top_k handled gracefully.")
    print("  [OK] Test 5 PASSED")

    # -------------------------------------------------------------
    # Test 6: Empty Collection (e.g. 'test' collection with 0 items)
    # -------------------------------------------------------------
    print("\n[Test 6] Empty Collection ('test' collection with 0 items)")
    empty_col_result = retriever.retrieve(Query(text="Linux", collection="test"))
    assert len(empty_col_result.chunks) == 0, "Empty collection must return 0 chunks"
    print("  [OK] Empty collection handled safely.")
    print("  [OK] Test 6 PASSED")

    # -------------------------------------------------------------
    # Test 7: RetrievalService Integration
    # -------------------------------------------------------------
    print("\n[Test 7] RetrievalService Integration")
    service = RetrievalService(retriever=retriever)
    service_result = service.retrieve(Query(text="Linux command", collection="linux", top_k=1))
    assert len(service_result.chunks) == 1, "RetrievalService must return 1 chunk"
    print(f"  [OK] RetrievalService successfully retrieved: {service_result.chunks[0].id}")
    print("  [OK] Test 7 PASSED")

    # -------------------------------------------------------------
    # Test 8: Collection Enum & Metadata Consistency
    # -------------------------------------------------------------
    print("\n[Test 8] Collection Enum Query (Collection.LINUX)")
    from knowledge.vector_db.collections import Collection
    enum_query = Query(text="Linux process", collection=Collection.LINUX, top_k=1)
    enum_result = retriever.retrieve(enum_query)
    assert len(enum_result.chunks) == 1, "Expected 1 chunk retrieved via Collection Enum"
    assert enum_result.chunks[0].metadata.get("collection") == "linux", "Expected chunk collection metadata to be 'linux'"
    print(f"  [OK] Collection Enum query returned chunk with consistent metadata: '{enum_result.chunks[0].metadata.get('collection')}'")
    print("  [OK] Test 8 PASSED")

    print("\n" + "=" * 60)
    print("ALL 8 KNOWLEDGE RETRIEVER TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")



if __name__ == "__main__":
    test_knowledge_retriever()
