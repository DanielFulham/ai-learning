from typing import Protocol


class InputProviderInterface(Protocol):
    """Seam for user input into the Auth workflow.

    Synchronous string prompting. prompt() takes the prompt text and
    returns the user's input as a str, echoing what is typed.
    prompt_secret() has the same shape but the typed input does not echo
    to the terminal — for password entry. Both return a plain str; the
    not-echoing behaviour is a concrete-implementation concern, not part
    of the type contract. Neither validates empty input; the caller owns
    whatever validation it needs.
    """

    def prompt(self, message: str) -> str: ...

    def prompt_secret(self, message: str) -> str: ...
