from knowledge.pipeline.ingestion_pipeline import IngestionPipeline

pipeline = IngestionPipeline()

document = pipeline.loader.load(
    "knowledge/documents/linux/Linux.pdf"
)

print("=" * 60)
print("DOCUMENT INFORMATION")
print("=" * 60)

print("Document Name :", document.name)
print("Document Type :", document.document_type)
print("Characters    :", len(document.content))

print("\nFirst 500 Characters:\n")
print(document.content[:500])

chunks = pipeline.chunker.chunk(document)

print("\n" + "=" * 60)
print("CHUNK INFORMATION")
print("=" * 60)

print("Chunks Created :", len(chunks))

if chunks:
    print("\nFirst Chunk Length :", len(chunks[0].text))