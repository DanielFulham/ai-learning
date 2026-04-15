# application/container.py
from infra.client import Client
from infra.embedding import Embedding
from application.book_search_service import BookSearchService
from infra.book_repository import BookRepository

def build_search_service(collection):
    return BookSearchService(collection)

def build_client():
    return Client().get_client()

def build_embedding():
    return Embedding().get_embedding_function()

def build_repository(client):
    return BookRepository(client)