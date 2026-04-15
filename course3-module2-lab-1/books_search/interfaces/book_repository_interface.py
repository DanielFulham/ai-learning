from typing import Protocol


class BookRepositoryInterface(Protocol):

    def create_collection(self, name: str, embedding_function) -> object:
        ...

    def add_books(self, collection, books: list) -> None:
        ...