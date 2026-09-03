from knowledge.storage.pdf_loader import PDFLoader
from knowledge.chunking.strategies import FixedSizeChunker
from knowledge.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder
)
from knowledge.vector_db.chroma_database import ChromaDatabase


class IngestionPipeline:

    def __init__(self):

        self.loader = PDFLoader()

        self.chunker = FixedSizeChunker()

        self.embedder = SentenceTransformerEmbedder()

        self.database = ChromaDatabase()

    def ingest(

        self,

        pdf_path: str,

        collection: str

    ):

        print("\nLoading document...")

        document = self.loader.load(pdf_path)

        print("Chunking...")

        chunks = self.chunker.chunk(document)

        print(f"{len(chunks)} chunks created.")

        print("Generating embeddings...")

        embeddings = self.embedder.embed_batch(chunks)

        print("Storing vectors...")

        for embedding in embeddings:

            self.database.add_embedding(

                collection,

                embedding

            )

        print("Done!")

        return document