from typing import Protocol


class LLMServiceInterface(Protocol):
    """Contract for a vision LLM service. Any class with this shape satisfies it."""

    def generate_response(
        self,
        encoded_image: str,
        user_query: str,
        assistant_prompt: str
    ) -> str:
        """Send base64 image + text prompt to a vision model, return raw text response."""
        ...