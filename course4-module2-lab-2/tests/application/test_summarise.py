import pytest
from unittest.mock import Mock, patch
from application.summarise import summarize_video, _create_summary_prompt

@pytest.fixture
def summarize_deps():
    return {
        "get_transcript": Mock(return_value=[Mock(text="Hello", start=0.0)]),
        "process": Mock(return_value="Text: Hello Start: 0.0"),
        "chunk_transcript": Mock(return_value=["chunk1", "chunk2"]),
        "llm": Mock(),
        "embedding_model": Mock(),
        "create_faiss_index": Mock(return_value=Mock()),
    }

def test_create_summary_prompt_has_correct_input_variables():
    prompt = _create_summary_prompt()
    assert prompt.input_variables == ["transcript"]

def test_create_summary_prompt_contains_transcript_placeholder():
    prompt = _create_summary_prompt()
    assert "{transcript}" in prompt.template

# --- summarize_video ---

def test_summarize_video_no_url_returns_error(summarize_deps):
    summary, _ = summarize_video(
        video_url="",
        state={},
        **summarize_deps
    )
    assert "valid YouTube" in summary

def test_summarize_video_populates_state(summarize_deps):

    state = {}
    summarize_video(
        video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        state=state,
        **summarize_deps
    )

    assert state.get("processed_transcript") == "Text: Hello Start: 0.0"

def test_summarize_video_returns_tuple(summarize_deps):
    result = summarize_video(
        video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        state={},
        **summarize_deps
    )

    assert isinstance(result, tuple)
    assert len(result) == 2