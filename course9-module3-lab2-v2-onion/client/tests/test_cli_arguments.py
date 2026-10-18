from cli_arguments import parse_arguments
from config import DEFAULT_ROOTS_DISPLAY, DEFAULT_SERVER


def test_with_no_arguments_both_default() -> None:
    assert parse_arguments(["gui_app.py"]) == (DEFAULT_SERVER, DEFAULT_ROOTS_DISPLAY)


def test_the_server_alone_keeps_the_default_roots_text() -> None:
    assert parse_arguments(["gui_app.py", "http://TEST_HOST:1"]) == (
        "http://TEST_HOST:1",
        DEFAULT_ROOTS_DISPLAY,
    )


def test_both_arguments_are_used_as_given() -> None:
    assert parse_arguments(["gui_app.py", "TEST_SERVER_SCRIPT.py", "TEST_ROOTS"]) == (
        "TEST_SERVER_SCRIPT.py",
        "TEST_ROOTS",
    )
