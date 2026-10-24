from unittest.mock import MagicMock

import gradio as gr
import pytest

from page_lifecycle import serve


def _make_page(order: list[str]) -> MagicMock:
    page = MagicMock(spec=gr.Blocks)
    page.queue.return_value = page
    page.close.side_effect = lambda: order.append("page")
    return page


def _interrupt() -> None:
    raise KeyboardInterrupt


def test_given_a_page_when_served_then_it_launches_without_blocking_on_the_host_and_port() -> None:
    order: list[str] = []
    page = _make_page(order)
    serve(page, lambda: order.append("session"), "TEST_HOST", 1234, wait=_interrupt)
    page.launch.assert_called_once_with(
        server_name="TEST_HOST", server_port=1234, prevent_thread_lock=True
    )


def test_given_a_stop_request_when_served_then_the_session_closes_before_the_page() -> None:
    """Pins F-L47-33: the MCP session must be closed while the page's loop is still running."""
    order: list[str] = []
    serve(_make_page(order), lambda: order.append("session"), "TEST_HOST", 1234, wait=_interrupt)
    assert order == ["session", "page"]


def test_given_the_wait_ends_without_an_interrupt_when_served_then_both_are_still_closed() -> None:
    order: list[str] = []
    serve(_make_page(order), lambda: order.append("session"), "TEST_HOST", 1234, wait=lambda: None)
    assert order == ["session", "page"]


def test_given_a_session_that_fails_to_close_when_served_then_the_page_still_closes() -> None:
    order: list[str] = []

    def failing_shutdown() -> None:
        raise RuntimeError("TEST_CLOSE_FAILED")

    with pytest.raises(RuntimeError, match="TEST_CLOSE_FAILED"):
        serve(_make_page(order), failing_shutdown, "TEST_HOST", 1234, wait=_interrupt)
    assert order == ["page"]


def test_given_a_page_that_cannot_launch_when_served_then_the_session_is_still_closed() -> None:
    order: list[str] = []
    page = _make_page(order)
    page.launch.side_effect = OSError("TEST_PORT_IN_USE")
    with pytest.raises(OSError, match="TEST_PORT_IN_USE"):
        serve(page, lambda: order.append("session"), "TEST_HOST", 1234, wait=_interrupt)
    assert order == ["session", "page"]
