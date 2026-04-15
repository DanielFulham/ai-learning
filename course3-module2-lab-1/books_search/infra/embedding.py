from chromadb.utils import embedding_functions
from interfaces.embedding_interface import EmbeddingInterface

class Embedding(EmbeddingInterface):

    def get_embedding_function(self):
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
