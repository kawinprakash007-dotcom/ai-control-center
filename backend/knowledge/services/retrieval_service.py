from knowledge.interfaces.retriever_interface import (
    RetrieverInterface,
)

from knowledge.models.query import Query
from knowledge.models.retrieval_result import RetrievalResult


class RetrievalService:

    def __init__(

        self,

        retriever: RetrieverInterface

    ):

        self.retriever = retriever

    def retrieve(

        self,

        query: Query

    ) -> RetrievalResult:

        return self.retriever.retrieve(query)