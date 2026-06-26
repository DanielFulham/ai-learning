from typing import Protocol


class InputProviderInterface(Protocol):
    """Seam for user input into the Auth workflow.

    Synchronous string prompting. The prompt() method takes the prompt
    text and returns the user's input as a str.

    V3b's C4 adds prompt_secret() for password input that doesn't
    echo. C3 uses prompt() only.
    """

    def prompt(self, message: str) -> str: ...
