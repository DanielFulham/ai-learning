from unittest.mock import patch
from infra.embedding import setup_embedding_model

def test_setup_embedding_model_returns_huggingface_embeddings():
    with patch("infra.embedding.HuggingFaceEmbeddings") as mock_embeddings:
        mock_instance = mock_embeddings.return_value

        result = setup_embedding_model()

        assert isinstance(result, type(mock_instance))
        mock_embeddings.assert_called_once_with(model_name="all-MiniLM-L6-v2")
