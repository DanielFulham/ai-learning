from dotenv import load_dotenv
import os
from infra.llm import initialize_llm
from infra.embedding import setup_embedding_model

load_dotenv()

def setup_credentials():
    return os.getenv("ANTHROPIC_API_KEY")

# Singletons — initialised once at startup
api_key = setup_credentials()
llm = initialize_llm(api_key)
embedding_model = setup_embedding_model()