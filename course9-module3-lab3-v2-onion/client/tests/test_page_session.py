import gradio as gr

from application.approval_desks import ApprovalDesks
from page_session import forget_session, session_id


def test_given_a_request_with_a_session_hash_when_read_then_that_hash_is_the_session_id() -> None:
    assert session_id(gr.Request(session_hash="TEST_HASH")) == "TEST_HASH"


def test_given_two_different_hashes_when_read_then_the_ids_differ() -> None:
    """Pins F-L47-18: two browser sessions never share an id, so never share a desk."""
    assert session_id(gr.Request(session_hash="TEST_A")) != session_id(
        gr.Request(session_hash="TEST_B")
    )


def test_given_a_request_with_no_session_hash_when_read_then_a_fixed_fallback_id_is_used() -> None:
    """Pins that a missing hash never creates a fresh desk per call, which would lose answers."""
    assert session_id(gr.Request()) == session_id(gr.Request())


def test_given_a_session_with_a_desk_when_the_page_closes_then_its_desk_is_forgotten() -> None:
    """Pins F-L47-27: the page's unload event drops the closed session's desk."""
    desks = ApprovalDesks()
    desks.desk_for("TEST_HASH")
    forget_session(desks, gr.Request(session_hash="TEST_HASH"))
    assert desks.count() == 0
