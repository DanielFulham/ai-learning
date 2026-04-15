from typing import Protocol

class EmbeddingInterface(Protocol):

    def get_embedding_function(self):
        ...