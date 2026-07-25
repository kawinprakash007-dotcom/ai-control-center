from pathlib import Path

from knowledge.models.document import Document

from knowledge.storage.document_loader import DocumentLoader


class ImageLoader(DocumentLoader):

    def load(self, path: str):

        file = Path(path)

        return Document(

            id=file.stem,

            name=file.name,

            path=str(file),

            content="",

            document_type="image",

            collection="vision"

        )

    def supported_extensions(self):

        return [

            ".png",

            ".jpg",

            ".jpeg",

            ".bmp",

            ".webp"

        ]