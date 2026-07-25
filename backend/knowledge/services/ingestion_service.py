from knowledge.models.document import Document


class IngestionService:

    """
    Responsible for loading documents into the Knowledge Engine.
    """

    def ingest_file(self, path: str) -> Document:
        """
        Load a single document.
        """

        raise NotImplementedError(
            "Document loading is not implemented yet."
        )

    def ingest_folder(self, path: str) -> list[Document]:
        """
        Load all supported documents from a folder.
        """

        raise NotImplementedError(
            "Folder ingestion is not implemented yet."
        )