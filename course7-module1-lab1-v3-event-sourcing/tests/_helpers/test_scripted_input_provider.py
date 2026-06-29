import pytest

from .scripted_input_provider import ScriptedInputProvider


class TestScriptedInputProviderPrompt:

    def test_prompt_returns_next_response(self) -> None:
        """Pinned: prompt() returns the next scripted response."""
        provider = ScriptedInputProvider(["alice"])

        assert provider.prompt("What is your username? ") == "alice"

    def test_prompt_returns_responses_in_order(self) -> None:
        """Pinned: successive prompt() calls drain the queue in order."""
        provider = ScriptedInputProvider(["first", "second"])

        assert provider.prompt("a") == "first"
        assert provider.prompt("b") == "second"

    def test_prompt_raises_indexerror_when_exhausted(self) -> None:
        """Pinned: an exhausted queue raises IndexError on prompt()."""
        provider = ScriptedInputProvider([])

        with pytest.raises(IndexError, match="exhausted at prompt:"):
            provider.prompt("a")


class TestScriptedInputProviderPromptSecret:

    def test_prompt_secret_returns_next_response(self) -> None:
        """Pinned: prompt_secret() draws from the same shared queue as
        prompt() — no separate secrets queue."""
        provider = ScriptedInputProvider(["secret"])

        assert provider.prompt_secret("Enter your password: ") == "secret"

    def test_prompt_and_prompt_secret_share_one_queue_in_order(self) -> None:
        """Pinned: prompt and prompt_secret interleaved consume the single
        queue in call order. The test author's model is 'the user types X
        then Y then Z'; whether the input echoed is a runtime concern."""
        provider = ScriptedInputProvider(["alice", "secret", "again"])

        assert provider.prompt("username? ") == "alice"
        assert provider.prompt_secret("password? ") == "secret"
        assert provider.prompt("username? ") == "again"

    def test_prompt_secret_raises_indexerror_when_exhausted(self) -> None:
        """Pinned: an exhausted queue raises IndexError on prompt_secret(),
        same failure shape as prompt()."""
        provider = ScriptedInputProvider([])

        with pytest.raises(IndexError, match="exhausted at prompt_secret:"):
            provider.prompt_secret("a")
