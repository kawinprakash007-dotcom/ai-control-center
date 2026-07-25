from pathlib import Path
import fitz

from knowledge.models.document import Document
from knowledge.storage.document_loader import DocumentLoader


class PDFLoader(DocumentLoader):

    def load(self, path: str) -> Document:

        file = Path(path).resolve()

        print(f"\nLooking for PDF at:\n{file}\n")

        if not file.exists():
            raise FileNotFoundError(
                f"PDF not found:\n{file}"
            )

        pdf = fitz.open(file)

        pages = []
        page_metadata = []

        for page_number, page in enumerate(pdf):

            text = page.get_text("text")

            pages.append(text)

            page_metadata.append(
                {
                    "page": page_number + 1,
                    "characters": len(text)
                }
            )

        pdf.close()

        return Document(
            id=file.stem,
            name=file.name,
            path=str(file),
            content="\n\n".join(pages),
            document_type="pdf",
            collection="general",
            metadata={
                "pages": len(pages),
                "page_metadata": page_metadata
            }
        )

    def supported_extensions(self):

        return [".pdf"]