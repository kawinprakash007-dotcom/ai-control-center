from pathlib import Path

from knowledge.models.document import Document

from knowledge.storage.document_loader import DocumentLoader


class CodeLoader(DocumentLoader):

    SUPPORTED = [

        ".py",

        ".cpp",

        ".c",

        ".java",

        ".js",

        ".ts",

        ".html",

        ".css",

        ".json"

    ]

    def load(self, path: str):

        file = Path(path)

        content = file.read_text(

            encoding="utf-8",

            errors="ignore"

        )

        return Document(

            id=file.stem,

            name=file.name,

            path=str(file),

            content=content,

            document_type="code",

            collection="coding"

        )

    def supported_extensions(self):

        return self.SUPPORTED