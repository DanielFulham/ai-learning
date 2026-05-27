"""
Configuration settings for the Style Finder application.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# Backend selection — set to False to use IBM watsonx
USE_LOCAL = False

# Model and API configuration
LLAMA_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "skills-network")
REGION = "us-south"

# Local model configuration
OLLAMA_MODEL_ID = "llava"

# Image processing settings
IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# Default similarity threshold
SIMILARITY_THRESHOLD = 0.8

# Number of alternatives to return from search
DEFAULT_ALTERNATIVES_COUNT = 5