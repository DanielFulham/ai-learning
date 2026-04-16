"""Configuration settings for the Icebreaker Bot."""

import os
from dotenv import load_dotenv

load_dotenv()

# IBM watsonx.ai settings
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")