from knowledge.models.embedding import Embedding

from knowledge.vector_db.chroma_database import ChromaDatabase


class VectorStore:

    def __init__(self):

        self.database = ChromaDatabase()

    def add(

        self,

        collection: str,

        embedding: Embedding

    ):

        self.database.add_embedding(

            collection,

            embedding

        )