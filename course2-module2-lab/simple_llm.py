# Import the necessary packages
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM
from dotenv import load_dotenv
import os
import gradio as gr

# Load in .env
load_dotenv()

project_id = os.getenv("WATSONX_PROJECT_ID")
# project_id="skills-network"  # If no api key used

# Specify the model
model_id = 'ibm/granite-4-h-small'

# Set the necessary parameters
parameters = {
    GenParams.MAX_NEW_TOKENS: 512,  # Specify the max tokens you want to generate
    GenParams.TEMPERATURE: 0.5, # This randomness or creativity of the model's responses
}

# Wrap up the model into WatsonxLLM inference
watsonx_llm = WatsonxLLM(
    model_id=model_id,
    url="https://us-south.ml.cloud.ibm.com",
    apikey=os.getenv("WATSONX_APIKEY"), # Comment out if using using lab
    project_id=project_id,
    params=parameters,
)

def generate_response(prompt_txt):
    response = watsonx_llm.invoke(prompt_txt)
    return response

chat_application = gr.Interface(
    fn=generate_response, 
    flagging_mode="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Watsonx.ai chatbot",
    description="Ask any question and the chatbot will try to answer."
)

chat_application.launch()