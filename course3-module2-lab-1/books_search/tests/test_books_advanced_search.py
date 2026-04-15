"""
Integration test for books_advanced_search.py

Runs the full pipeline against an in-memory ChromaDB instance.
Note: This is a local integration test — requires chromadb and 
sentence-transformers installed in the venv. Not suitable for CI 
without those dependencies available.
"""

import pytest
import chromadb
from application.book_loader import load_books
from application.book_document_builder import build_book_documents, build_book_metadatas


@pytest.fixture
def collection():
    client = chromadb.Client()
    from chromadb.utils import embedding_functions
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.create_collection(
        name="test_book_collection",
        metadata={"description": "Integration test collection"},
        configuration={
            "hnsw": {"space": "cosine"},
            "embedding_function": ef,
        }
    )
    books = load_books()
    collection.add(
        ids=[book.id for book in books],
        documents=build_book_documents(books),
        metadatas=build_book_metadatas(books)
    )
    yield collection
    client.delete_collection(name="test_book_collection")


def test_collection_has_correct_document_count(collection):
    all_items = collection.get()
    assert len(all_items['documents']) == 8


def test_similarity_search_returns_results(collection):
    results = collection.query(
        query_texts=["magical fantasy adventure"],
        n_results=3
    )
    assert len(results['ids'][0]) == 3


def test_metadata_filter_by_genre_returns_correct_books(collection):
    results = collection.get(
        where={"genre": {"$in": ["Fantasy", "Science Fiction"]}}
    )
    assert len(results['ids']) == 4


def test_metadata_filter_by_rating_returns_correct_books(collection):
    results = collection.get(
        where={"rating": {"$gte": 4.5}}
    )
    assert len(results['ids']) == 2


def test_combined_search_returns_dystopian_books(collection):
    results = collection.query(
        query_texts=["dystopian society surveillance"],
        n_results=5,
        where={
            "$and": [
                {"genre": "Dystopian"},
                {"rating": {"$gte": 4.0}}
            ]
        }
    )
    assert len(results['ids'][0]) >= 1
    genres = [m['genre'] for m in results['metadatas'][0]]
    assert all(g == "Dystopian" for g in genres)