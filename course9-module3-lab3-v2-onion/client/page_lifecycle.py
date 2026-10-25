import time
from collections.abc import Callable

import gradio as gr


def _wait_until_interrupted() -> None:
    # A short sleep in a loop, because a long blocking wait is not interrupted by Ctrl+C on Windows.
    while True:
        time.sleep(0.5)


def serve(
    page: gr.Blocks,
    shutdown: Callable[[], None],
    host: str,
    port: int,
    wait: Callable[[], None] = _wait_until_interrupted,
) -> None:
    """Run a page until it is stopped, then close the MCP session and the page, in that order.

    Gradio's own blocking launch closes the page on Ctrl+C but knows nothing about the MCP
    session, so the stdio server could be left to outlive the app. The session is closed first,
    while the page's event loop is still running, because the session lives on that loop.
    """
    try:
        page.queue().launch(server_name=host, server_port=port, prevent_thread_lock=True)
        wait()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            shutdown()
        finally:
            page.close()
