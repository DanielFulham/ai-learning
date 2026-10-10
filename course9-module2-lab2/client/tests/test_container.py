import asyncio
from unittest.mock import MagicMock

from application.container import initialise
from interfaces.chat_model_interface import ChatModelInterface
from interfaces.mcp_session_interface import McpSessionInterface
from interfaces.user_input_interface import UserInputInterface


def test_injected_parts_are_wired_together() -> None:
    session = MagicMock(spec=McpSessionInterface)
    user_input = MagicMock(spec=UserInputInterface)
    user_input.ask.return_value = "q"

    service = initialise(
        chat_model=MagicMock(spec=ChatModelInterface), session=session, user_input=user_input
    )
    asyncio.run(service.run())

    session.connect.assert_awaited_once()
    session.close.assert_awaited_once()


def test_defaults_build_without_a_key_or_a_server() -> None:
    service = initialise("TEST_SERVER_SCRIPT.py")
    assert hasattr(service, "run")
