from typing import Protocol


class BookDocumentBuilderInterface(Protocol):

    def build_book_documents(self, books: list) -> list:
        ...

    def build_book_metadatas(self, books: list) -> list:
        ...