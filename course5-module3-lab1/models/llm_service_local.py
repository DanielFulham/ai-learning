"""
Local Ollama service for interacting with LLaVA vision model.
"""

import logging
import ollama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LlamaVisionService:
    """
    Provides methods to interact with a local vision model via Ollama.
    """

    def __init__(self, model_id="llava", project_id=None, region=None,
                 temperature=0.2, top_p=0.6, api_key=None, max_tokens=2000):
        self.model_id = model_id
        self.params = {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens
        }

    def generate_response(self, encoded_image, prompt):
        try:
            logger.info("Sending request to Ollama with prompt length: %d", len(prompt))

            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encoded_image]
                }
            ]

            response = ollama.chat(model=self.model_id, messages=messages, options=self.params)
            content = response['message']['content']

            logger.info("Received response with length: %d", len(content))

            return content

        except Exception as e:
            logger.error("Error generating response: %s", str(e))
            return f"Error generating response: {e}"

    def generate_fashion_response(self, user_image_base64, matched_row, all_items,
                                  similarity_score, threshold=0.8):
        items_list = []
        for _, row in all_items.iterrows():
            item_str = f"{row['Item Name']} ({row['Price']}): {row['Link']}"
            items_list.append(item_str)

        items_description = "\n".join([f"- {item}" for item in items_list])

        if similarity_score >= threshold:
            assistant_prompt = (
                f"You're conducting a professional retail catalog analysis. "
                f"This image shows standard clothing items available in department stores. "
                f"Focus exclusively on professional fashion analysis for a clothing retailer. "
                f"ITEM DETAILS (always include this section in your response):\n{items_description}\n\n"
                "Please:\n"
                "1. Identify and describe the clothing items objectively (colors, patterns, materials)\n"
                "2. Categorize the overall style (business, casual, etc.)\n"
                "3. Include the ITEM DETAILS section at the end\n\n"
                "This is for a professional retail catalog. Use formal, clinical language."
            )
        else:
            assistant_prompt = (
                f"You're conducting a professional retail catalog analysis. "
                f"This image shows standard clothing items available in department stores. "
                f"Focus exclusively on professional fashion analysis for a clothing retailer. "
                f"SIMILAR ITEMS (always include this section in your response):\n{items_description}\n\n"
                "Please:\n"
                "1. Note these are similar but not exact items\n"
                "2. Identify clothing elements objectively (colors, patterns, materials)\n"
                "3. Include the SIMILAR ITEMS section at the end\n\n"
                "This is for a professional retail catalog. Use formal, clinical language."
            )

        response = self.generate_response(encoded_image=user_image_base64, prompt=assistant_prompt)

        if len(response) < 100:
            logger.info("Response appears incomplete, creating basic response")
            section_header = "ITEM DETAILS:" if similarity_score >= threshold else "SIMILAR ITEMS:"
            response = f"# Fashion Analysis\n\nThis outfit features a collection of carefully coordinated pieces.\n\n{section_header}\n{items_description}"

        if "ITEM DETAILS" not in response and "SIMILAR ITEMS" not in response and "Similar Items" not in response:
            logger.info("Item details section missing from response")
            section_header = "ITEM DETAILS:" if similarity_score >= threshold else "SIMILAR ITEMS:"
            response += f"\n\n{section_header}\n{items_description}"

        return response