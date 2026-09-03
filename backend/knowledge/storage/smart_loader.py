from pathlib import Path

from knowledge.models.document import Document

from knowledge.storage.pdf_loader import PDFLoader
from knowledge.storage.ocr_loader import OCRLoader
from knowledge.storage.text_loader import TextLoader


class SmartLoader:

    """
    Universal Document Loader.

    Automatically selects the correct loader.

    Supported:

    • PDF
    • TXT

    (Markdown, Code and Images will be added later.)
    """

    def __init__(self):

        self.pdf_loader = PDFLoader()

        self.ocr_loader = OCRLoader()

        self.text_loader = TextLoader()

        # Minimum extracted characters
        # below this we assume scanned PDF

        self.ocr_threshold = 100

    def load(self, path: str) -> Document:

        file = Path(path)

        extension = file.suffix.lower()

        # -------------------------
        # PDF
        # -------------------------

        if extension == ".pdf":

            document = self.pdf_loader.load(path)

            if len(document.content.strip()) >= self.ocr_threshold:

                print("✓ Searchable PDF detected.")

                return document

            print("✓ Image PDF detected.")

            print("Switching to OCR...")

            return self.ocr_loader.load(path)

        # -------------------------
        # TXT
        # -------------------------

        if extension == ".txt":

            return self.text_loader.load(path)

        raise ValueError(

            f"Unsupported document type: {extension}"

        )