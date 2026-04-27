from langchain_huggingface import HuggingFaceEmbeddings

def setup_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
