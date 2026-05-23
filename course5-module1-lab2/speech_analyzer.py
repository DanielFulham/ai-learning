import gradio as gr
from langchain_ollama import OllamaLLM # For LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from transformers import pipeline  # For Speech-to-Text

#######------------- LLM Initialization-------------#######

# Initialize the local LLM using OllamaLLM
llm = OllamaLLM(
    model="llama3.2",
    temperature=0.2,
    num_predict=1024,
)

#######------------- Helper Functions-------------#######

# Function to remove non-ASCII characters
def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

def product_assistant(ascii_transcript):
    system_prompt = """You are an intelligent assistant specializing in financial products.
    Your task is to process transcripts of earnings calls, ensuring that all references to
    financial products and common financial terms are in the correct format. For each
    financial product or common term that is typically abbreviated as an acronym, the full term
    should be spelled out followed by the acronym in parentheses. For example, '401k' should be
    transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 
    'Health Savings Account (HSA)', 'ROA' should be transformed to 'Return on Assets (ROA)'.
    In cases where numerical figures do not represent specific financial products, leave them as is.
    Produce the adjusted transcript only, no commentary."""

    prompt_input = system_prompt + "\n" + ascii_transcript

    cleanup_llm = OllamaLLM(
        model="llama3.2",
        temperature=0.2,
        num_predict=1024,
    )
    
    return cleanup_llm.invoke(prompt_input)

#######------------- Prompt Template and Chain-------------#######

# Define the prompt template
template = """
Generate meeting minutes and a list of tasks based on the provided context.

Context:
{context}

Meeting Minutes:
- Key points discussed
- Decisions made

Task List:
- Actionable items with assignees and deadlines
"""

prompt = ChatPromptTemplate.from_template(template)

# Define the chain
chain = (
    prompt | llm | StrOutputParser()
)

#######------------- Speech2text and Pipeline-------------#######

# Speech-to-text pipeline
def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",  # back to tiny
        chunk_length_s=30,
    )

    raw_transcript = pipe(audio_file, batch_size=8)["text"] # type: ignore
    ascii_transcript = remove_non_ascii(raw_transcript)

    adjusted_transcript = product_assistant(ascii_transcript)
    result = chain.invoke({"context": adjusted_transcript})

    # Write the result to a file for downloading
    output_file = "meeting_minutes_and_tasks.txt"
    with open(output_file, "w") as file:
        file.write(result)

    # Return the textual result and the file for download
    return result, output_file

#######------------- Gradio Interface-------------#######

audio_input = gr.Audio(sources="upload", type="filepath", label="Upload your audio file")
output_text = gr.Textbox(label="Meeting Minutes and Tasks")
download_file = gr.File(label="Download the Generated Meeting Minutes and Tasks")

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=[output_text, download_file],
    title="AI Meeting Assistant",
    description="Upload an audio file of a meeting. This tool will transcribe the audio, fix product-related terminology, and generate meeting minutes along with a list of tasks."
)

iface.launch()
