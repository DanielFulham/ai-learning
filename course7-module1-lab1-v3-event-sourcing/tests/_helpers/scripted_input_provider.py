"""Test helper: ScriptedInputProvider.

Returns pre-scripted prompt responses in order. Used in tests that need
deterministic input without patching builtins.input or constructing a
full ConsoleInputProvider. C4 will add prompt_secret() when F06 lands.
"""


class ScriptedInputProvider:
    """Returns pre-scripted responses in order. For tests."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def prompt(self, message: str) -> str:
        if not self._responses:
            raise IndexError(
                f"ScriptedInputProvider exhausted at prompt: {message}"
            )
        return self._responses.pop(0)
