import pytest
from unittest.mock import Mock
from application.book_search_service import BookSearchService

@pytest.fixture
def mock_collection():
    return Mock()


@pytest.fixture
def book_search_service(mock_collection):
    return BookSearchService(mock_collection)


def test_search_by_similarity_calls_query_with_correct_params(book_search_service, mock_collection):
    mock_collection.query.return_value = {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
    query_text = "adventure"
    n_results = 5
    
    book_search_service.search_by_similarity(query_text, n_results)
    
    mock_collection.query.assert_called_once_with(
        query_texts=[query_text],
        n_results=n_results
    )


def test_filter_by_genre_calls_get_with_correct_params(book_search_service, mock_collection):
    mock_collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    genres = ["Fiction", "Mystery"]
    
    book_search_service.filter_by_genre(genres)
    
    mock_collection.get.assert_called_once_with(
        where={"genre": {"$in": genres}}
    )


def test_filter_by_rating_calls_get_with_correct_params(book_search_service, mock_collection):
    mock_collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    min_rating = 4.5
    
    book_search_service.filter_by_rating(min_rating)
    
    mock_collection.get.assert_called_once_with(
        where={"rating": {"$gte": min_rating}}
    )


def test_search_combined_calls_query_with_correct_params(book_search_service, mock_collection):
    mock_collection.query.return_value = {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
    query_text = "mystery"
    genre = "Fiction"
    min_rating = 4.0
    
    book_search_service.search_combined(query_text, genre, min_rating)
    
    mock_collection.query.assert_called_once_with(
        query_texts=[query_text],
        n_results=5,
        where={
            "$and": [
                {"genre": genre},
                {"rating": {"$gte": min_rating}}
            ]
        }
    )


def test_filter_by_decade_calls_get_with_correct_params(book_search_service, mock_collection):
    mock_collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    start_year = 1990
    end_year = 1999
    
    book_search_service.filter_by_decade(start_year, end_year)
    
    mock_collection.get.assert_called_once_with(
        where={
            "$and": [
                {"year": {"$gte": start_year}},
                {"year": {"$lt": end_year}}
            ]
        }
    )


def test_filter_by_page_count_calls_get_with_correct_params(book_search_service, mock_collection):
    mock_collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    target = 300
    tolerance = 50
    
    book_search_service.filter_by_page_count(target, tolerance)
    
    mock_collection.get.assert_called_once_with(
        where={
            "$and": [
                {"pages": {"$gte": target - tolerance}},
                {"pages": {"$lte": target + tolerance}}
            ]
        }
    )


def test_search_by_themes_calls_query_with_correct_params(book_search_service, mock_collection):
    mock_collection.query.return_value = {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
    query_text = "magic"
    themes = ["fantasy", "adventure"]
    
    book_search_service.search_by_themes(query_text, themes)
    
    mock_collection.query.assert_called_once()
    call_args = mock_collection.query.call_args
    assert call_args.kwargs["query_texts"] == [query_text]
    assert call_args.kwargs["n_results"] == 5
    assert "$or" in call_args.kwargs["where"]

# Unhappy Paths

def test_search_by_similarity_with_empty_results(book_search_service, mock_collection):
    mock_collection.query.return_value = {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
    query_text = "nonexistent theme"
    n_results = 5
    
    book_search_service.search_by_similarity(query_text, n_results)
    
    mock_collection.query.assert_called_once_with(
        query_texts=[query_text],
        n_results=n_results
    )

def test_filter_by_genre_with_no_matching_books(book_search_service, mock_collection):
    mock_collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    genres = ["Nonexistent Genre"]
    
    book_search_service.filter_by_genre(genres)
    
    mock_collection.get.assert_called_once_with(
        where={"genre": {"$in": genres}}
    )

def test_filter_by_rating_with_no_matching_books(book_search_service, mock_collection):
    mock_collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    min_rating = 5.0
    
    book_search_service.filter_by_rating(min_rating)
    
    mock_collection.get.assert_called_once_with(
        where={"rating": {"$gte": min_rating}}
    )

def test_search_combined_with_no_results(book_search_service, mock_collection):
    mock_collection.query.return_value = {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
    query_text = "nonexistent theme"
    genre = "Nonexistent Genre"
    min_rating = 5.0
    
    book_search_service.search_combined(query_text, genre, min_rating)
    
    mock_collection.query.assert_called_once_with(
        query_texts=[query_text],
        n_results=5,
        where={
            "$and": [
                {"genre": genre},
                {"rating": {"$gte": min_rating}}
            ]
        }
    )

def test_filter_by_decade_with_no_books_in_range(book_search_service, mock_collection):
    mock_collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    start_year = 1800
    end_year = 1809
    book_search_service.filter_by_decade(start_year, end_year)
    mock_collection.get.assert_called_once_with(
        where={
            "$and": [
                {"year": {"$gte": start_year}},
                {"year": {"$lt": end_year}}
            ]
        }
    )

