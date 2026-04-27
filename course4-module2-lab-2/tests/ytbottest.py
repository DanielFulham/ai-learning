import sys
from unittest.mock import Mock, patch

mock_config = Mock()
mock_config.llm = Mock()
mock_config.embedding_model = Mock()
sys.modules["config"] = mock_config

from ytbot import handle_summarize, handle_answer_question

def test_handle_summarize_calls_summarize_video():
    with patch("ytbot.summarize_video") as mock_summarize:
        mock_summarize.return_value = ("summary", {})
        _ = handle_summarize("https://www.youtube.com/watch?v=dQw4w9WgXcQ", {})
        mock_summarize.assert_called_once()

def test_handle_answer_question_calls_answer_question():
    with patch("ytbot.answer_question") as mock_answer:
        mock_answer.return_value = ("answer", {})
        _ = handle_answer_question(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "What is RAG?",
            {}
        )
        mock_answer.assert_called_once()