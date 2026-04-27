import pytest
from unittest.mock import Mock
from application.qa import answer_question, _create_qa_prompt_template

@pytest.fixture
def qa_deps():
    return {
        "get_transcript": Mock(return_value=[Mock(text="Hello", start=0.0)]),
        "process": Mock(return_value="Text: Hello Start: 0.0"),
        "chunk_transcript": Mock(return_value=["chunk1", "chunk2"]),
        "llm": Mock(),
        "embedding_model": Mock(),
        "create_faiss_index": Mock(return_value=Mock()),
        "retrieve": Mock(return_value=["relevant chunk"]),
    }

# --- _create_qa_prompt_template ---

def test_create_qa_prompt_has_correct_input_variables():
    prompt = _create_qa_prompt_template()
    assert "context" in prompt.input_variables
    assert "question" in prompt.input_variables

def test_create_qa_prompt_contains_placeholders():
    prompt = _create_qa_prompt_template()
    assert "{context}" in prompt.template
    assert "{question}" in prompt.template

# --- answer_question ---
def test_answer_question_no_url_no_transcript_returns_error(qa_deps):
    state = {}
    answer, _ = answer_question(
        video_url="",
        user_question="What is RAG?",
        state=state,
        **qa_deps
    )

    assert "valid YouTube" in answer

def test_answer_question_fetches_transcript_if_missing(qa_deps):
    state = {}
    answer_question(
        video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        user_question="What is RAG?",
        state=state,
        **qa_deps
    )

    qa_deps["get_transcript"].assert_called_once_with("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

def test_answer_question_skips_fetch_if_transcript_in_state(qa_deps):
    state = {
        "processed_transcript": "Text: Hello Start: 0.0",
        "faiss_index": Mock(),
    }
    
    answer_question(
        video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        user_question="What is RAG?",
        state=state,
        **qa_deps
    )

    qa_deps["get_transcript"].assert_not_called()

def test_answer_question_returns_tuple(qa_deps):
    state = {
        "processed_transcript": "Text: Hello Start: 0.0",
        "faiss_index": Mock()
    }
    result = answer_question(
        video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        user_question="What is RAG?",
        state=state,
        **qa_deps
    )
    assert isinstance(result, tuple)
    assert len(result) == 2