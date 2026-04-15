from typing import Protocol


class BookLoaderInterface(Protocol):

    def load_books(self) -> list:
        ...