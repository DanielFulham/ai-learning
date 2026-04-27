# Import necessary libraries for the YouTube bot
import gradio as gr
from infra.vector_store import create_faiss_index, retrieve
from config import llm, embedding_model
from domain.transcript import get_transcript, process, chunk_transcript
from application.summarise import summarize_video
from application.qa import answer_question

def handle_summarize(video_url, state):
    return summarize_video(
        video_url=video_url,
        get_transcript=get_transcript,
        process=process,
        chunk_transcript=chunk_transcript,
        llm=llm,
        embedding_model=embedding_model,
        create_faiss_index=create_faiss_index,
        state=state
    )

def handle_answer_question(video_url, user_question, state):
    return answer_question(
        video_url=video_url,
        user_question=user_question,
        state=state,
        get_transcript=get_transcript,
        process=process,
        chunk_transcript=chunk_transcript,
        llm=llm,
        embedding_model=embedding_model,
        create_faiss_index=create_faiss_index,
        retrieve=retrieve
    )

with gr.Blocks() as interface:
    gr_state = gr.State({})

    # Input field for YouTube URL
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube video URL here...")
    
    # Outputs for summary and answer
    summary_output = gr.Textbox(label="Video Summary", placeholder="The summary of the video will appear here...", lines=5)
    question_input = gr.Textbox(label="Ask a Question about the Video", placeholder="Enter your question here...")
    answer_output = gr.Textbox(label="Answer to Your Question", placeholder="The answer to your question will appear here...", lines=5)

    # Buttons for selecting functionalities after fetching transcript
    summarize_button = gr.Button("Summarize Video")
    question_button = gr.Button("Ask Question")

    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    # Set up button actions
    summarize_button.click(fn=handle_summarize, inputs=[video_url, gr_state], outputs=[summary_output, gr_state])
    question_button.click(fn=handle_answer_question, inputs=[video_url, question_input, gr_state], outputs=[answer_output, gr_state])

# Launch the app with specified server name and port
interface.launch(server_name="127.0.0.1", server_port=7860, share=False)
