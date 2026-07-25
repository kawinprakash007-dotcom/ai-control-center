from knowledge.storage.pdf_loader import PDFLoader
from knowledge.chunking.strategies import FixedSizeChunker
from knowledge.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder
)

loader = PDFLoader()

document = loader.load(
    "knowledge/documents/linux/Linux.pdf"
)

chunker = FixedSizeChunker()

chunks = chunker.chunk(document)

embedder = SentenceTransformerEmbedder()

embedding = embedder.embed(chunks[0])

print("=" * 50)
print("EMBEDDING TEST")
print("=" * 50)

print("Vector Length:", len(embedding.vector))

print("First 10 Values:")

print(embedding.vector[:10])