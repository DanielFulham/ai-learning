from unittest.mock import MagicMock

import gradio as gr

from application.interfaces.gui_service_interface import GuiServiceInterface
from gui_app import build_interface, with_choices


def test_with_choices_keeps_the_text_and_fills_the_dropdown() -> None:
    output, update = with_choices("TEST_OUTPUT", ["TEST_A", "TEST_B"])
    assert output == "TEST_OUTPUT"
    assert update["choices"] == ["TEST_A", "TEST_B"]


def test_with_no_choices_the_dropdown_is_emptied() -> None:
    assert with_choices("", [])[1]["choices"] == []


def test_the_page_builds_without_launching() -> None:
    page = build_interface(MagicMock(spec=GuiServiceInterface), "http://TEST_HOST", "TEST_ROOTS")
    assert isinstance(page, gr.Blocks)
    assert page.title == "MCP HTTP Client"
