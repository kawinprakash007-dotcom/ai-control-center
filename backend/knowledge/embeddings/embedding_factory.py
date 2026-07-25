from knowledge.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)


class EmbeddingFactory:

    @staticmethod
    def create():

        return SentenceTransformerEmbedder()