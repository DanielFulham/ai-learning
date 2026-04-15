import pytest
from unittest.mock import Mock
from infra.book_repository import BookRepository


@pytest.fixture
def mock_client():
    return Mock()


@pytest.fixture
def repository(mock_client):
    return BookRepository(client=mock_client)


def test_create_collection_called_with_correct_config(repository, mock_client):
    mock_ef = Mock()

    repository.create_collection("book_collection", mock_ef)

    mock_client.create_collection.assert_called_once_with(
        name="book_collection",
        metadata={"description": "A collection for storing book data"},
        configuration={
            "hnsw": {"space": "cosine"},
            "embedding_function": mock_ef,
        }
    )


def test_add_books_calls_collection_add(repository):
    from domain.book import Book

    mock_collection = Mock()
    book = Book(
        id="book_1",
        title="1984",
        author="George Orwell",
        genre="Dystopian",
        year=1949,
        rating=4.4,
        pages=328,
        description="A chilling vision of totalitarian control and surveillance society",
        themes="totalitarianism, surveillance, freedom, truth",
        setting="Oceania, dystopian future"
    )

    repository.add_books(mock_collection, [book])

    mock_collection.add.assert_called_once_with(
        ids=["book_1"],
        documents=[book.to_document()],
        metadatas=[book.to_metadata()]
    )

def test_create_collection_raises_on_duplicate_name(repository, mock_client):
    mock_client.create_collection.side_effect = Exception("Collection already exists")

    with pytest.raises(Exception) as exc_info:
        repository.create_collection("existing_collection", Mock())

    assert "Collection already exists" in str(exc_info.value)

def test_add_books_with_empty_list(repository, mock_client):
    mock_collection = Mock()

    repository.add_books(mock_collection, [])

    mock_collection.add.assert_not_called()