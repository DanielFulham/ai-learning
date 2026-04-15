from interfaces.book_repository_interface import BookRepositoryInterface
from application.book_document_builder import build_book_documents, build_book_metadatas

class BookRepository(BookRepositoryInterface):

    def __init__(self, client):
        self.client = client

    def create_collection(self, name: str, embedding_function) -> object:
        return self.client.create_collection(
            name=name,
            metadata={"description": "A collection for storing book data"},
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": embedding_function,
            }
        )

    def add_books(self, collection, books: list) -> None:
        collection.add(
            ids=[book.id for book in books],
            documents=build_book_documents(books),
            metadatas=build_book_metadatas(books)
        )