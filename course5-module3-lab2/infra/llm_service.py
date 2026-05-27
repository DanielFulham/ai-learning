import os
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

import config


class WatsonxLLMService:
    """IBM watsonx implementation. Calls Meta's Llama 4 Maverick via watsonx hosted API."""

    def __init__(self):
        credentials = Credentials(
            url=config.WATSONX_URL,
            api_key=os.getenv("WATSONX_API_KEY")
        )
        self.client = APIClient(credentials)

        self.model = ModelInference(
            model_id=config.WATSONX_MODEL_ID,
            credentials=credentials,
            project_id=os.getenv("WATSONX_PROJECT_ID"),
            params=TextChatParameters(
                temperature=config.WATSONX_TEMPERATURE,
                top_p=config.WATSONX_TOP_P,
                max_tokens=config.WATSONX_MAX_TOKENS,
            )
        )

    def generate_response(self, encoded_image, user_query, assistant_prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": assistant_prompt + "\n\n" + user_query},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
                ]
            }
        ]
        response = self.model.chat(messages=messages)
        return response['choices'][0]['message']['content']