from knowledge.interfaces.vector_database_interface import (
    VectorDatabaseInterface,
)

from knowledge.models.document import Document


class IndexingService:

    def __init__(

        self,

        vector_database: VectorDatabaseInterface

    ):

        self.vector_database = vector_database

    def index_document(

        self,

        document: Document

    ):

        raise NotImplementedError(
            "Document indexing is not implemented yet."
        )

    def index_documents(

        self,

        documents: list[Document]

    ):

        for document in documents:

            self.index_document(document)