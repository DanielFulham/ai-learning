from langchain_core.prompts import PromptTemplate

def answer_question(video_url, user_question, state,
                    get_transcript, process, chunk_transcript,
                    llm, embedding_model, create_faiss_index, retrieve):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    if not state.get("processed_transcript"):
        if video_url:
            fetched_transcript = get_transcript(video_url)
            state["processed_transcript"] = process(fetched_transcript)
            chunks = chunk_transcript(state["processed_transcript"])
            state["faiss_index"] = create_faiss_index(chunks, embedding_model)
            state["embedding_model"] = embedding_model
        else:
            return "Please provide a valid YouTube URL.", state

    if state.get("faiss_index") is None:
        chunks = chunk_transcript(state["processed_transcript"])
        state["faiss_index"] = create_faiss_index(chunks, embedding_model)
        state["embedding_model"] = embedding_model

    if state.get("processed_transcript") and user_question:
        qa_prompt = _create_qa_prompt_template()
        qa_chain = _create_qa_chain(llm, qa_prompt)
        answer = _generate_answer(user_question, state["faiss_index"], qa_chain, retrieve)
        return answer, state
    else:
        return "Please provide a valid question and ensure the transcript has been fetched.", state

def _create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.

    Returns:
        PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    """
    
    # Define the template string
    qa_template = """
    You are an expert assistant providing detailed answers based on the following video content.

    Relevant Video Context: {context}

    Based on the above context, please answer the following question:
    Question: {question}
    """

    # Create the PromptTemplate object
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )

    return prompt_template

def _create_qa_chain(llm, prompt_template):
    """
    Create an LLMChain for question answering.

    Args:
        llm: Language model instance
            The language model to use in the chain (e.g., WatsonxGranite).
        prompt_template: PromptTemplate
            The prompt template to use for structuring inputs to the language model.

    Returns:
        LLMChain: An instantiated LLMChain ready for question answering.
    """
    return prompt_template | llm

def _generate_answer(question, faiss_index, qa_chain, retrieve, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.

    Args:
        question: str
            The user's question.
        faiss_index: FAISS
            The FAISS index containing the embedded documents.
        qa_chain: LLMChain
            The question-answering chain (LLMChain) to use for generating answers.
        retrieve: function
            The function to use for retrieving relevant context from the FAISS index.
        k: int, optional (default=3)
            The number of relevant documents to retrieve.

    Returns:
        str: The generated answer to the user's question.
    """
    # Retrieve relevant context from the FAISS index
    relevant_context = retrieve(question, faiss_index, k=k)
        
    # Generate an answer using the question-answering chain with the retrieved context
    answer = qa_chain.invoke({"context": relevant_context, "question": question}).content
    
    return answer