from domain.book import Book
from data.books import books as raw_books


def load_books():
    return [Book(**b) for b in raw_books]