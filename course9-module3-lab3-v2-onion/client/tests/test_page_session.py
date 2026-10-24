import gradio as gr

from page_session import session_id


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
