from typing import Protocol


class InputProviderInterface(Protocol):
    """Provider seam for user input.

    Concretes back this with stdin (`ConsoleInputProvider`), a canned list
    (`ScriptedInputProvider`), or any future source — pipe, HTTP body,
    Slack DM. The auth workflow's `input_node` depends only on this
    Protocol; `input()` never appears in the application layer.

    Closes V1 finding #5 — moving stdin behind an interface.
    """

    def prompt(self, message: str) -> str: ...