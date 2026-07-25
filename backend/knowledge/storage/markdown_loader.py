from pathlib import Path

from knowledge.models.document import Document

from knowledge.storage.document_loader import DocumentLoader


class MarkdownLoader(DocumentLoader):

    def load(self, path: str) -> Document:

        file = Path(path)

        content = file.read_text(
            encoding="utf-8"
        )

        return Document(

            id=file.stem,

            name=file.name,

            path=str(file),

            content=content,

            document_type="markdown",

            collection="general"

        )

    def supported_extensions(self):

        return [

            ".md"

        ]