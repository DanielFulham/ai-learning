from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from application.llm_text import invoke_text


def test_returns_stripped_string_for_text_response() -> None:
    model = MagicMock(spec=BaseChatModel)
    model.invoke.return_value = AIMessage(content="  the answer with whitespace  ")
    result = invoke_text(model, "any prompt")
    assert result == "the answer with whitespace"


def test_passes_prompt_through_to_model_invoke() -> None:
    model = MagicMock(spec=BaseChatModel)
    model.invoke.return_value = AIMessage(content="answer")
    invoke_text(model, "Context: x\nQuestion: y")
    model.invoke.assert_called_once_with("Context: x\nQuestion: y")


def test_non_str_content_raises_type_error() -> None:
    """A multimodal model returning a content-block list trips the guard
    rather than silently returning a stringified list."""
    model = MagicMock(spec=BaseChatModel)
    model.invoke.return_value = AIMessage(
        content=[{"type": "text", "text": "block one"}]
    )
    with pytest.raises(TypeError, match="got list"):
        invoke_text(model, "any prompt")