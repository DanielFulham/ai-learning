import pytest
from application.book_loader import load_books
from application.book_document_builder import build_book_documents, build_book_metadatas
from domain.book import Book


@pytest.fixture
def sample_book():
    return Book(
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


def test_build_book_documents_returns_list_of_strings(sample_book):
    documents = build_book_documents([sample_book])

    assert isinstance(documents, list)
    assert len(documents) == 1
    assert isinstance(documents[0], str)


def test_build_book_documents_contains_expected_content(sample_book):
    documents = build_book_documents([sample_book])

    assert "1984" in documents[0]
    assert "George Orwell" in documents[0]
    assert "Dystopian" in documents[0]
    assert "totalitarianism, surveillance, freedom, truth" in documents[0]


def test_build_book_metadatas_returns_list_of_dicts(sample_book):
    metadatas = build_book_metadatas([sample_book])

    assert isinstance(metadatas, list)
    assert len(metadatas) == 1
    assert isinstance(metadatas[0], dict)


def test_build_book_metadatas_contains_expected_fields(sample_book):
    metadatas = build_book_metadatas([sample_book])

    assert metadatas[0]['title'] == "1984"
    assert metadatas[0]['author'] == "George Orwell"
    assert metadatas[0]['genre'] == "Dystopian"
    assert metadatas[0]['year'] == 1949
    assert metadatas[0]['rating'] == 4.4
    assert metadatas[0]['pages'] == 328


def test_build_book_metadatas_excludes_id(sample_book):
    metadatas = build_book_metadatas([sample_book])

    assert metadatas[0].get('id') is None