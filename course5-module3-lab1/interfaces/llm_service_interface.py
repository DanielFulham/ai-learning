"""
Interface contract for LLM vision services.
"""

from typing import Protocol
import pandas as pd


class LLMServiceInterface(Protocol):

    def generate_response(self, encoded_image: str, prompt: str) -> str:
        ...

    def generate_fashion_response(
        self,
        user_image_base64: str,
        matched_row: pd.Series,
        all_items: pd.DataFrame,
        similarity_score: float,
        threshold: float = 0.8
    ) -> str:
        ...