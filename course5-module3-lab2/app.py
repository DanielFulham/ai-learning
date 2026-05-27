import re
import base64
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash

import config
from interfaces.llm_service_interface import LLMServiceInterface
from infra.llm_service import WatsonxLLMService
from infra.llm_service_local import OllamaLLMService

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)


def create_llm_service() -> LLMServiceInterface:
    """Composition root — picks the backend based on config.USE_LOCAL."""
    if config.USE_LOCAL:
        print("🦙 Using local Ollama LLaVA")
        return OllamaLLMService()
    print("☁️  Using IBM watsonx Llama 4 Maverick")
    return WatsonxLLMService()


llm_service = create_llm_service()


def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        encoded_image = base64.b64encode(bytes_data).decode("utf-8")
        return encoded_image
    else:
        raise FileNotFoundError("No file uploaded")


def format_response(response_text):
    response_text = re.sub(r"\*\*(.*?)\*\*", r"<p><strong>\1</strong></p>", response_text)
    response_text = re.sub(r"(?m)^\s*\*\s(.*)", r"<li>\1</li>", response_text)
    response_text = re.sub(r"(<li>.*?</li>)+", lambda m: f"<ul>{m.group(0)}</ul>", response_text, flags=re.DOTALL)
    response_text = re.sub(r"</p>(?=<p>)", r"</p><br>", response_text)
    response_text = re.sub(r"(\n|\\n)+", r"<br>", response_text)
    return response_text


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form.get("user_query")
        uploaded_file = request.files.get("file")

        if uploaded_file:
            encoded_image = input_image_setup(uploaded_file)

            if not encoded_image:
                flash("Error processing the image. Please try again.", "danger")
                return redirect(url_for("index"))

            assistant_prompt = """
            You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide a detailed nutritional assessment using the following format:

        1. **Identification**: List each identified food item clearly, one per line.
        2. **Portion Size & Calorie Estimation**: For each identified food item, specify the portion size and provide an estimated number of calories. Use bullet points with the following structure:
        - **[Food Item]**: [Portion Size], [Number of Calories] calories

        Example:
        *   **Salmon**: 6 ounces, 210 calories
        *   **Asparagus**: 3 spears, 25 calories

        3. **Total Calories**: Provide the total number of calories for all food items.

        Example:
        Total Calories: [Number of Calories]

        4. **Nutrient Breakdown**: Include a breakdown of key nutrients such as **Protein**, **Carbohydrates**, **Fats**, **Vitamins**, and **Minerals**. Use bullet points, and for each nutrient provide details about the contribution of each food item.

        Example:
        *   **Protein**: Salmon (35g), Asparagus (3g), Tomatoes (1g) = [Total Protein]

        5. **Health Evaluation**: Evaluate the healthiness of the meal in one paragraph.

        6. **Disclaimer**: Include the following exact text as a disclaimer:

        The nutritional information and calorie estimates provided are approximate and are based on general food data. 
        Actual values may vary depending on factors such as portion size, specific ingredients, preparation methods, and individual variations. 
        For precise dietary advice or medical guidance, consult a qualified nutritionist or healthcare provider.

        Format your response exactly like the template above to ensure consistency.

        """

            try:
                raw = llm_service.generate_response(encoded_image, user_query, assistant_prompt)
                response = format_response(raw)
            except Exception as e:
                print(f"Error in generating response: {e}")
                response = "<p>An error occurred while generating the response.</p>"

            return render_template("index.html", user_query=user_query, response=response)

        else:
            flash("Please upload an image file.", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)