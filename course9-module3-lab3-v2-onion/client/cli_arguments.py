from pathlib import Path

from config import DEFAULT_SERVER_PYTHON, DEFAULT_SERVER_SCRIPT
from domain.server_launch import ServerLaunch


def parse_arguments(argv: list[str]) -> ServerLaunch:
    """The server script and the interpreter to launch it with. Both are optional.

    The lab requires the script; here it defaults to this repo's V2 server, started with that
    server's own virtual environment.
    """
    script = Path(argv[1]) if len(argv) > 1 else DEFAULT_SERVER_SCRIPT
    python = Path(argv[2]) if len(argv) > 2 else DEFAULT_SERVER_PYTHON
    return ServerLaunch(script=script, python=python)
