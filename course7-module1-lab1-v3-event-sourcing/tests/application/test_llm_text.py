from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from application.llm_text import invoke_text


class TestInvokeText:

    def test_returns_stripped_str_content(self) -> None:
        model = MagicMock(spec=BaseChatModel)
        model.invoke.return_value = AIMessage(content="  An answer.  ")

        result = invoke_text(model, "Q")

        assert result == "An answer."

    def test_non_str_content_raises_type_error(self) -> None:
        """Pinned: the narrowing guard at the LLM-output boundary. A
        multimodal model returning a list of content blocks would
        silently pass V2's `response.content` without this check;
        the guard makes the boundary loud."""
        model = MagicMock(spec=BaseChatModel)
        model.invoke.return_value = AIMessage(content=[{"type": "text", "text": "x"}])

        with pytest.raises(TypeError, match="Expected str content"):
            invoke_text(model, "Q")