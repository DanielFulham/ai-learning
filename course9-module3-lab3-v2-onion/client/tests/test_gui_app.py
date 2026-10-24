from unittest.mock import MagicMock

import gradio as gr

from application.approval_desks import ApprovalDesks
from application.interfaces.approval_desk_interface import ApprovalDeskInterface
from application.interfaces.gui_service_interface import GuiServiceInterface
from domain.approval import ApprovalChoice
from gui_app import answer_waiting_question, build_interface, waiting_question_text, with_choices


def test_given_text_and_names_when_filling_a_dropdown_then_the_text_stays_and_choices_are_set() -> (
    None
):
    output, update = with_choices("TEST_OUTPUT", ["TEST_A", "TEST_B"])
    assert output == "TEST_OUTPUT"
    assert update["choices"] == ["TEST_A", "TEST_B"]


def test_given_no_names_when_filling_a_dropdown_then_it_is_emptied() -> None:
    assert with_choices("", [])[1]["choices"] == []


def test_given_no_waiting_question_when_shown_then_the_text_says_none_is_waiting() -> None:
    desk = MagicMock(spec=ApprovalDeskInterface)
    desk.pending_message.return_value = None
    assert waiting_question_text(desk) == "No question is waiting."


def test_given_a_waiting_question_when_shown_then_the_text_is_the_servers_question() -> None:
    desk = MagicMock(spec=ApprovalDeskInterface)
    desk.pending_message.return_value = "TEST_QUESTION"
    assert waiting_question_text(desk) == "TEST_QUESTION"


def test_given_a_waiting_question_when_the_user_clicks_approve_then_the_desk_gets_accept() -> None:
    desk = MagicMock(spec=ApprovalDeskInterface)
    desk.respond.return_value = True
    text = answer_waiting_question(desk, ApprovalChoice.ACCEPT)
    desk.respond.assert_called_once_with(ApprovalChoice.ACCEPT)
    assert text == "Answer sent: accept"


def test_given_nothing_waiting_when_the_user_clicks_a_button_then_it_says_so() -> None:
    """Pins that a stray click is told nothing happened, never silently ignored."""
    desk = MagicMock(spec=ApprovalDeskInterface)
    desk.respond.return_value = False
    assert answer_waiting_question(desk, ApprovalChoice.ACCEPT) == "No question is waiting."


def test_given_two_sessions_when_one_answers_then_the_other_desk_is_untouched() -> None:
    """Pins F-L47-18 for the GUI: a click in one session cannot answer another's question."""
    desks = ApprovalDesks()
    first, second = desks.desk_for("TEST_A"), desks.desk_for("TEST_B")
    assert answer_waiting_question(first, ApprovalChoice.ACCEPT) == "No question is waiting."
    assert second.pending_message() is None


def test_given_a_gui_service_when_the_page_is_built_then_it_builds_without_launching() -> None:
    page = build_interface(MagicMock(spec=GuiServiceInterface), ApprovalDesks())
    assert isinstance(page, gr.Blocks)
    assert page.title == "MCP Permission Client"


def test_given_the_gui_page_when_it_is_built_then_closing_the_page_is_handled() -> None:
    """Pins F-L47-27: the page registers an unload handler so a closed tab frees its desk."""
    page = build_interface(MagicMock(spec=GuiServiceInterface), ApprovalDesks())
    assert any(("unload" in [t[1] for t in fn.targets]) for fn in page.fns.values())
