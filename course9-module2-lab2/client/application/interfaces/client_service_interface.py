from typing import Protocol


class ClientServiceInterface(Protocol):
    """What the entry point needs from the client application."""

    async def run(self) -> None:
        """Connect to the server, run the menu until the person quits, then disconnect."""
        ...
