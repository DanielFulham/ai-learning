import ollama
import config


class OllamaLLMService:
    """Local Ollama implementation. Calls LLaVA on local GPU."""

    def __init__(self):
        self.model_id = config.OLLAMA_MODEL_ID
        self.params = config.OLLAMA_PARAMS

    def generate_response(self, encoded_image, user_query, assistant_prompt):
        messages = [
            {
                "role": "user",
                "content": assistant_prompt + "\n\n" + user_query,
                "images": [encoded_image]   # base64 as top-level field, not in content array
            }
        ]
        response = ollama.chat(
            model=self.model_id,
            messages=messages,
            options=self.params
        )
        return response['message']['content']