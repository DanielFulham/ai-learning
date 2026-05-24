import base64
from pathlib import Path
import webbrowser
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.images.generate(
    model="gpt-image-1",
    prompt="a beautiful lake with a sunset",
    size="1024x1024",
    n=1,
)

image_base64 = response.data[0].b64_json
if image_base64:
    image_bytes = base64.b64decode(image_base64)
    output_path = Path("beautiful_lake_with_sunset-gpt-image-1.png")
    output_path.write_bytes(image_bytes)
    webbrowser.open(str(output_path.resolve()))
else:
    print("No image returned")