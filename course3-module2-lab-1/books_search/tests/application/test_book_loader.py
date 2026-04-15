from application.book_loader import load_books
from domain.book import Book


def test_load_books_returns_list_of_books():
    books = load_books()

    assert isinstance(books, list)
    assert len(books) == 8
    assert all(isinstance(book, Book) for book in books)


def test_load_books_first_book_is_great_gatsby():
    books = load_books()

    assert books[0].id == "book_1"
    assert books[0].title == "The Great Gatsby"
    assert books[0].author == "F. Scott Fitzgerald"
    assert books[0].genre == "Classic"
    assert books[0].year == 1925
    assert books[0].rating == 4.1
    assert books[0].pages == 180
    assert books[0].description == "A tragic tale of wealth, love, and the American Dream in the Jazz Age"
    assert books[0].themes == "wealth, corruption, American Dream, social class"
    assert books[0].setting == "New York, 1920s"

