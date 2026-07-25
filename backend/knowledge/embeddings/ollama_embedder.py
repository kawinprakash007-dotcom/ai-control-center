from knowledge.interfaces.embedder_interface import EmbedderInterface


class OllamaEmbedder(EmbedderInterface):

    def embed(self, chunk):

        raise NotImplementedError()

    def embed_batch(self, chunks):

        raise NotImplementedError()