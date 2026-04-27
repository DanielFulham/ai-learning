import pytest
from unittest.mock import Mock, patch
from infra.vector_store import create_faiss_index, retrieve

@pytest.fixture
def mock_faiss_index():
    index = Mock()
    index.similarity_search.return_value = ["doc1", "doc2", "doc3"]
    return index

# --- create_faiss_index ---

def test_create_faiss_index_returns_index():
    chunks = ["This is a test chunk.", "Another test chunk."]
    embedding_model = Mock()

    with patch("infra.vector_store.FAISS.from_texts") as mock_faiss:
            mock_index = Mock()
            mock_faiss.return_value = mock_index
            
            result = create_faiss_index(chunks, embedding_model)
            
            assert result == mock_index
            mock_faiss.assert_called_once_with(chunks, embedding_model)

# --- retrieve ---

def test_retrieve_returns_list(mock_faiss_index):
    result = retrieve("test query", mock_faiss_index)

    assert isinstance(result, list)
    assert len(result) == 3

def test_retrieve_calls_similarity_search_with_correct_args(mock_faiss_index):
    retrieve("test query", mock_faiss_index, k=5)

    mock_faiss_index.similarity_search.assert_called_once_with("test query", k=5)

def test_retrieve_default_k_is_7(mock_faiss_index):
    retrieve("test query", mock_faiss_index)

    mock_faiss_index.similarity_search.assert_called_once_with("test query", k=7)