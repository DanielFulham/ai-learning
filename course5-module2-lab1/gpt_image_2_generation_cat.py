import base64
from pathlib import Path
import webbrowser
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.images.generate(
    model="gpt-image-2",
    prompt="a white siamese cat",
    size="1024x1024",
    n=1,
)

image_base64 = response.data[0].b64_json
if image_base64:
    image_bytes = base64.b64decode(image_base64)
    output_path = Path("siamese_cat-gpt-image-2.png")
    output_path.write_bytes(image_bytes)
    webbrowser.open(str(output_path.resolve()))
else:
    print("No image returned")