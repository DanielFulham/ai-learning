import ollama
import base64
import requests

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


def generate_model_response(encoded_image, user_query, assistant_prompt="You are a helpful assistant. Answer the following user query in 1 or 2 sentences: "):
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
            "content": assistant_prompt + user_query,
            "images": [encoded_image]
        }
    ]
    response = ollama.chat(model=model_id, messages=messages, options=params)
    return response['message']['content']


url_image_1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5uo16pKhdB1f2Vz7H8Utkg/image-1.png'
url_image_2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fsuegY1q_OxKIxNhf6zeYg/image-2.png'
url_image_3 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KCh_pM9BVWq_ZdzIBIA9Fw/image-3.png'
url_image_4 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VaaYLw52RaykwrE3jpFv7g/image-4.png'

image_urls = [url_image_1, url_image_2, url_image_3, url_image_4]

model_id = "llava:latest"

params = {
    "temperature": 0.2,
    "top_p": 0.5,
}

encoded_images = encode_images_to_base64(image_urls)

# --- Image Captioning ---
print("=== Image Captioning ===\n")
user_query = "Describe the photo"
for i in range(len(encoded_images)):
    image = encoded_images[i]
    response = generate_model_response(image, user_query)
    print(f"Description for image {i + 1}: {response}\n")

# --- Object Detection: car count ---
print("=== Object Detection ===\n")
image = encoded_images[1]
user_query = "How many cars are in this image?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))
print()

# --- Damage Assessment ---
print("=== Damage Assessment ===\n")
image = encoded_images[2]
user_query = "How severe is the damage in this image?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))
print()

# --- Nutrition Label Reading ---
print("=== Nutrition Label Reading ===\n")
image = encoded_images[3]
user_query = "How much sodium is in this product?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))
print()

# --- Exercises ---
print("=== Exercise 1: Cholesterol ===\n")
image = encoded_images[3]
user_query = "How much cholesterol is in this product?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))
print()

print("=== Exercise 2: Jacket Colour ===\n")
image = encoded_images[1]
user_query = "What is the color of the woman's jacket?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))