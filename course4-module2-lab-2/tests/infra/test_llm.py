from unittest.mock import patch
from infra.llm import initialize_llm

def test_initialize_llm_returns_chat_anthropic():
    with patch("infra.llm.ChatAnthropic") as mock_chat_anthropic:
        mock_instance = mock_chat_anthropic.return_value

        result = initialize_llm("test-api-key")

        assert isinstance(result, type(mock_instance))
        mock_chat_anthropic.assert_called_once_with(
            model="claude-haiku-4-5",
            api_key="test-api-key",
            max_tokens=1000
        )