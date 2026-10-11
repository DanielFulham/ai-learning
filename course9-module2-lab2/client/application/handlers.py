from application.formatters import format_progress
from interfaces.mcp_session_interface import (
    ElicitationCallback,
    NotificationCallback,
    ProgressCallback,
)
from interfaces.user_input_interface import UserInputInterface


def make_elicitation_handler(user_input: UserInputInterface) -> ElicitationCallback:
    """Answer a server's request for input by asking the person at the keyboard."""

    async def handle(message: str, fields: list[str]) -> dict[str, str] | None:
        user_input.show(f"Server asks: {message}")
        values: dict[str, str] = {}
        for field_name in fields:
            answer = user_input.ask(f"Enter value for '{field_name}': ")
            if not answer:
                return None
            values[field_name] = answer
        return values

    return handle


def make_progress_handler(user_input: UserInputInterface) -> ProgressCallback:
    """Show the server's progress updates."""

    async def handle(progress: float, total: float | None, message: str | None) -> None:
        user_input.show(format_progress(progress, total, message))

    return handle


def make_notification_handler(user_input: UserInputInterface) -> NotificationCallback:
    """Show the server's notifications."""

    async def handle(method: str) -> None:
        user_input.show(f"Received: {method}")
        if method == "notifications/tools/list_changed":
            user_input.show("Tools have changed - might want to refresh tool cache")
        elif method == "notifications/resources/list_changed":
            user_input.show("Resources have changed")

    return handle
