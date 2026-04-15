import pytest
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

def test_book_creation(sample_book):
    assert sample_book.title == "1984"
    assert sample_book.author == "George Orwell"
    assert sample_book.genre == "Dystopian"
    assert sample_book.year == 1949
    assert sample_book.rating == 4.4
    assert sample_book.pages == 328
    assert sample_book.description == "A chilling vision of totalitarian control and surveillance society"
    assert sample_book.themes == "totalitarianism, surveillance, freedom, truth"
    assert sample_book.setting == "Oceania, dystopian future"
    assert sample_book.id == "book_1"

def test_book_to_document(sample_book):
    document = sample_book.to_document()
    expected_start = "1984 with the description A chilling vision of totalitarian control and surveillance society explores themes of totalitarianism, surveillance, freedom, truth and is set in Oceania, dystopian future. It is a Dystopian book published in 1949. Written by George Orwell."
    assert document.startswith(expected_start)

def test_book_to_metadata(sample_book):
    metadata = sample_book.to_metadata()
    assert metadata['title'] == "1984"
    assert metadata['author'] == "George Orwell"
    assert metadata['genre'] == "Dystopian"
    assert metadata['year'] == 1949
    assert metadata['rating'] == 4.4
    assert metadata['pages'] == 328
    assert metadata['description'] == "A chilling vision of totalitarian control and surveillance society"
    assert metadata['themes'] == "totalitarianism, surveillance, freedom, truth"
    assert metadata['setting'] == "Oceania, dystopian future"
    assert metadata.get('id') is None  # ID should not be in metadata