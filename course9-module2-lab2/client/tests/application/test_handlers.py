import asyncio
from collections.abc import Awaitable
from unittest.mock import MagicMock

from application.handlers import (
    make_elicitation_handler,
    make_notification_handler,
    make_progress_handler,
)
from interfaces.user_input_interface import UserInputInterface


async def _resolve[T](awaitable: Awaitable[T]) -> T:
    return await awaitable


def _make_user_input(*answers: str) -> MagicMock:
    user_input = MagicMock(spec=UserInputInterface)
    user_input.ask.side_effect = list(answers)
    return user_input


def _shown(user_input: MagicMock) -> list[str]:
    return [c.args[0] for c in user_input.show.call_args_list]


def test_elicitation_collects_each_field() -> None:
    user_input = _make_user_input("TEST_A", "TEST_B")
    handler = make_elicitation_handler(user_input)
    result = asyncio.run(_resolve(handler("TEST_MSG", ["file_path", "name"])))
    assert result == {"file_path": "TEST_A", "name": "TEST_B"}
    assert _shown(user_input) == ["Server asks: TEST_MSG"]


def test_elicitation_declines_on_an_empty_answer() -> None:
    handler = make_elicitation_handler(_make_user_input("TEST_A", ""))
    assert asyncio.run(_resolve(handler("TEST_MSG", ["file_path", "name"]))) is None


def test_progress_handler_shows_formatted_progress() -> None:
    user_input = _make_user_input()
    asyncio.run(_resolve(make_progress_handler(user_input)(5, 20, "TEST_MSG")))
    assert _shown(user_input) == ["Progress: 25.0% - TEST_MSG"]


def test_notification_handler_explains_list_changes() -> None:
    user_input = _make_user_input()
    asyncio.run(_resolve(make_notification_handler(user_input)("notifications/tools/list_changed")))
    assert _shown(user_input) == [
        "Received: notifications/tools/list_changed",
        "Tools have changed - might want to refresh tool cache",
    ]


def test_notification_handler_other_methods_are_just_named() -> None:
    user_input = _make_user_input()
    asyncio.run(_resolve(make_notification_handler(user_input)("notifications/message")))
    assert _shown(user_input) == ["Received: notifications/message"]
