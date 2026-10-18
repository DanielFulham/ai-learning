from config import DEFAULT_ROOTS_DISPLAY, DEFAULT_SERVER


def parse_arguments(argv: list[str]) -> tuple[str, str]:
    """The server (an http(s) address or a server script) and the roots text for the page header.

    Both are optional. The lab requires both; here they default to the lab server's address and
    "workspace".
    """
    server = argv[1] if len(argv) > 1 else DEFAULT_SERVER
    roots = argv[2] if len(argv) > 2 else DEFAULT_ROOTS_DISPLAY
    return server, roots
