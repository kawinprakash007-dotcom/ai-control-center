from knowledge.storage.pdf_loader import PDFLoader
from knowledge.chunking.strategies import FixedSizeChunker

loader = PDFLoader()

document = loader.load(
    "knowledge/documents/linux/Linux.pdf"
)

chunker = FixedSizeChunker(
    chunk_size=1000,
    overlap=200
)

chunks = chunker.chunk(document)

print("=" * 50)
print("CHUNKER TEST")
print("=" * 50)

print("Chunks Created:", len(chunks))

print("\nFirst Chunk\n")
print(chunks[0].text[:300])

print("\nLast Chunk\n")
print(chunks[-1].text[:300])