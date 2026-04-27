from langchain_core.prompts import PromptTemplate

def summarize_video(video_url, get_transcript, process, chunk_transcript,
                    llm, embedding_model, create_faiss_index, state):
    """
    Title: Summarize YouTube Video Content

    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """

    if video_url:
        # Fetch and preprocess the transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
        state["processed_transcript"] = processed_transcript

    else:
        return "Please provide a valid YouTube video URL to fetch the transcript.", state
    
    if processed_transcript:
        state["embedding_model"] = embedding_model
        chunks = chunk_transcript(processed_transcript)
        faiss_index = create_faiss_index(chunks, embedding_model)
        state["faiss_index"] = faiss_index

        # Step 4: Create the summary prompt and chain
        summary_prompt = _create_summary_prompt()
        summary_chain = _create_summary_chain(llm, summary_prompt)

        # Step 5: Generate the video summary
        summary = summary_chain.invoke({"transcript": processed_transcript}).content
        return summary, state

    else:
        return "No transcript available. Please fetch the transcript first.", state
    
def _create_summary_prompt():
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.
    
    :return: PromptTemplate object
    """
    # Define the template for the summary prompt
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    # Create the PromptTemplate object with the defined template
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )
    
    return prompt

def _create_summary_chain(llm, prompt):
    """
    Create an LLMChain for generating summaries.
    
    :param llm: Language model instance
    :param prompt: PromptTemplate instance
    :return: LLMChain instance
    """
    return prompt | llm