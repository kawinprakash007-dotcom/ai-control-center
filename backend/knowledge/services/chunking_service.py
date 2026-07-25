from knowledge.chunking.chunker import Chunker

from knowledge.models.document import Document
from knowledge.models.chunk import Chunk


class ChunkingService:

    def __init__(

        self,

        chunker: Chunker

    ):

        self.chunker = chunker

    def chunk_document(

        self,

        document: Document

    ) -> list[Chunk]:

        return self.chunker.chunk(document)