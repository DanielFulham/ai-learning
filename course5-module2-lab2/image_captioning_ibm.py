import base64
import requests
from dotenv import load_dotenv
import os
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

load_dotenv()

# --- Credentials ---
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key=os.getenv("WATSONX_API_KEY"),
)

project_id = os.getenv("WATSONX_PROJECT_ID")

# --- Model ---
model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"

params = TextChatParameters(
    temperature=0.2,
    top_p=0.5,
)

model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params
)

# --- Image URLs ---
url_image_1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5uo16pKhdB1f2Vz7H8Utkg/image-1.png'
url_image_2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fsuegY1q_OxKIxNhf6zeYg/image-2.png'
url_image_3 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KCh_pM9BVWq_ZdzIBIA9Fw/image-3.png'
url_image_4 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VaaYLw52RaykwrE3jpFv7g/image-4.png'

image_urls = [url_image_1, url_image_2, url_image_3, url_image_4]


# --- Encode Images ---
def encode_images_to_base64(image_urls):
    """
    Downloads and encodes a list of image URLs to base64 strings.

    Parameters:
    - image_urls (list): A list of image URLs.

    Returns:
    - list: A list of base64-encoded image strings.
    """
    encoded_images = []
    for url in image_urls:
        response = requests.get(url)
        if response.status_code == 200:
            encoded_image = base64.b64encode(response.content).decode("utf-8")
            encoded_images.append(encoded_image)
            print(type(encoded_image))
        else:
            print(f"Warning: Failed to fetch image from {url} (Status code: {response.status_code})")
            encoded_images.append(None)
    return encoded_images


# --- Inference Function ---
DEFAULT_PROMPT = "You are a helpful assistant. Answer the following user query in 1 or 2 sentences: "

def generate_model_response(encoded_image, user_query, assistant_prompt=DEFAULT_PROMPT):
    """
    Sends an image and a query to the model and retrieves the description or answer.

    Parameters:
    - encoded_image (str): Base64-encoded image string.
    - user_query (str): The user's question about the image.
    - assistant_prompt (str): Optional prompt to guide the model's response.

    Returns:
    - str: The model's response for the given image and query.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": assistant_prompt + user_query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + encoded_image,
                    }
                }
            ]
        }
    ]
    response = model.chat(messages=messages)
    return response['choices'][0]['message']['content']


# --- Run ---
encoded_images = encode_images_to_base64(image_urls)

# Image Captioning
print("=== Image Captioning ===\n")
user_query = "Describe the photo"
for i in range(len(encoded_images)):
    response = generate_model_response(encoded_images[i], user_query)
    print(f"Description for image {i + 1}: {response}\n")

# Object Detection
print("=== Object Detection ===\n")
user_query = "How many cars are in this image?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(encoded_images[1], user_query))
print()

# Damage Assessment
print("=== Damage Assessment ===\n")
user_query = "How severe is the damage in this image?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(encoded_images[2], user_query))
print()

# Nutrition Label
print("=== Nutrition Label Reading ===\n")
user_query = "How much sodium is in this product?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(encoded_images[3], user_query))
print()

# Exercise 1
print("=== Exercise 1: Cholesterol ===\n")
user_query = "How much cholesterol is in this product?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(encoded_images[3], user_query))
print()

# Exercise 2
print("=== Exercise 2: Jacket Colour ===\n")
user_query = "What is the color of the woman's jacket?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(encoded_images[1], user_query))