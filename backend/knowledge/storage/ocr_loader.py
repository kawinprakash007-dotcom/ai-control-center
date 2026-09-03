from pathlib import Path
import tempfile
import os

import fitz
import easyocr

from knowledge.models.document import Document
from knowledge.storage.document_loader import DocumentLoader


class OCRLoader(DocumentLoader):
    """
    OCR Loader using EasyOCR.

    Supports:
    - Scanned PDFs
    - Image PDFs
    - Certificates
    - Books
    """

    def __init__(self):

        self.reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

    def load(self, path: str) -> Document:

        file = Path(path)

        pdf = fitz.open(file)

        pages = []

        page_metadata = []

        with tempfile.TemporaryDirectory() as temp_dir:

            for page_number, page in enumerate(pdf):

                image_path = os.path.join(
                    temp_dir,
                    f"page_{page_number}.png"
                )

                pix = page.get_pixmap(
                    dpi=300
                )

                pix.save(image_path)

                result = self.reader.readtext(
                    image_path,
                    detail=0,
                    paragraph=True
                )

                page_text = "\n".join(result)

                pages.append(page_text)

                page_metadata.append(
                    {
                        "page": page_number + 1,
                        "characters": len(page_text)
                    }
                )

        pdf.close()

        content = "\n\n".join(pages)

        return Document(

            id=file.stem,

            name=file.name,

            path=str(file),

            content=content,

            document_type="pdf",

            collection="general",

            metadata={
                "pages": len(pages),
                "ocr": True,
                "page_metadata": page_metadata
            }
        )

    def supported_extensions(self):

        return [".pdf"]