from uuid import uuid4

from knowledge.chunking.chunker import Chunker

from knowledge.models.chunk import Chunk
from knowledge.models.document import Document


class FixedSizeChunker(Chunker):

    def __init__(

        self,

        chunk_size: int = 800,

        overlap: int = 150

    ):

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        document: Document
    ) -> list[Chunk]:

        chunks = []

        text = document.content

        start = 0

        index = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk_text = text[start:end]

            chunks.append(

                Chunk(

                    id=str(uuid4()),

                    document_id=document.id,

                    text=chunk_text,

                    chunk_index=index

                )

            )

            start += self.chunk_size - self.overlap

            index += 1

        return chunks