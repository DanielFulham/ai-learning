from typing import Protocol


class UserInputInterface(Protocol):
    """How the client talks to the person at the keyboard."""

    def ask(self, prompt: str) -> str:
        """Show the prompt and return the person's reply, without surrounding spaces."""
        ...

    def show(self, text: str) -> None:
        """Show a line of text to the person."""
        ...
