from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(

        self,

        model_name: str = "BAAI/bge-base-en-v1.5"

    ):

        self.model = SentenceTransformer(model_name)

    def encode(

        self,

        text: str

    ):

        return self.model.encode(

            text,

            normalize_embeddings=True

        )