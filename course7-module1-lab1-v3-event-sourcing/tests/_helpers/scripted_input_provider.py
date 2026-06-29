"""Test helper: ScriptedInputProvider.

Returns pre-scripted prompt responses in order. Used in tests that need
deterministic input without patching builtins.input or constructing a
full ConsoleInputProvider. prompt() and prompt_secret() draw from one
shared queue in call order — the test author's model is "the user types
X then Y then Z"; whether the input echoed or not is a runtime concern,
not a test concern.
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

    def prompt_secret(self, message: str) -> str:
        if not self._responses:
            raise IndexError(
                f"ScriptedInputProvider exhausted at prompt_secret: {message}"
            )
        return self._responses.pop(0)
