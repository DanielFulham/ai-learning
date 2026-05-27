# Backend selection
USE_LOCAL = False   # True = Ollama LLaVA, False = IBM watsonx

# IBM watsonx
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
WATSONX_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"

# Watsonx generation params
WATSONX_TEMPERATURE = 0.7    # balanced default; 0.0 not actually deterministic on hosted infra
WATSONX_TOP_P = 1.0
WATSONX_MAX_TOKENS = 2000

# Ollama local
OLLAMA_MODEL_ID = "llava:latest"
OLLAMA_PARAMS = {"temperature": 0.2, "top_p": 0.6, "num_predict": 2000}