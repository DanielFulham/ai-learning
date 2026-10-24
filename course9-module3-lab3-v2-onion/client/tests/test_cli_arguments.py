from pathlib import Path

from cli_arguments import parse_arguments
from config import DEFAULT_SERVER_PYTHON, DEFAULT_SERVER_SCRIPT
from domain.server_launch import ServerLaunch


def test_given_no_arguments_when_parsed_then_both_default_to_the_v2_server() -> None:
    assert parse_arguments(["gui_app.py"]) == ServerLaunch(
        DEFAULT_SERVER_SCRIPT, DEFAULT_SERVER_PYTHON
    )


def test_given_a_script_alone_when_parsed_then_the_interpreter_stays_the_default() -> None:
    launch = parse_arguments(["gui_app.py", "TEST_SERVER.py"])
    assert launch == ServerLaunch(Path("TEST_SERVER.py"), DEFAULT_SERVER_PYTHON)


def test_given_both_arguments_when_parsed_then_both_are_used_as_given() -> None:
    launch = parse_arguments(["gui_app.py", "TEST_SERVER.py", "TEST_PYTHON.exe"])
    assert launch == ServerLaunch(Path("TEST_SERVER.py"), Path("TEST_PYTHON.exe"))


def test_given_the_default_script_when_checked_then_it_is_the_server_folders_entry_point() -> None:
    """Pins the layout this client relies on: ../server/run_server.py beside this folder."""
    assert DEFAULT_SERVER_SCRIPT.name == "run_server.py"
    assert DEFAULT_SERVER_SCRIPT.parent.name == "server"
