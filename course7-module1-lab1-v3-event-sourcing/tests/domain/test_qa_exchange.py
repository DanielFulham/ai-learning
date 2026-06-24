from dataclasses import FrozenInstanceError, replace

import pytest

from domain.error_info import ErrorInfo
from domain.qa_exchange import QAExchange


def _make_exchange(
    question: str = "What is LangGraph?",
    context: str | None = None,
    answer: str | None = None,
    error_info: ErrorInfo | None = None,
) -> QAExchange:
    return QAExchange(
        question=question,
        context=context,
        answer=answer,
        error_info=error_info,
    )


class TestQAExchangeV2InvariantPreserved:
    """The V2 'question is non-empty after strip' invariant survives V3a's
    additive field. Adding error_info must not weaken the invariant — these
    tests pin the V2 contract still holds under the new shape.
    """

    def test_empty_question_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty after strip"):
            QAExchange(question="")

    def test_whitespace_only_question_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty after strip"):
            QAExchange(question="   ")

    def test_question_with_content_is_accepted(self) -> None:
        exchange = QAExchange(question="What is LangGraph?")
        assert exchange.question == "What is LangGraph?"


class TestQAExchangeV3aErrorInfo:

    def test_error_info_defaults_to_none(self) -> None:
        """Pinned: error_info is an additive field with a None default. V2
        construction sites that don't pass it continue to work unchanged."""
        exchange = _make_exchange()
        assert exchange.error_info is None

    def test_error_info_can_be_set(self) -> None:
        info = ErrorInfo(
            exception_type="OllamaConnectionError",
            exception_message="Connection refused",
        )
        exchange = _make_exchange(error_info=info)
        assert exchange.error_info == info

    def test_answer_and_error_info_can_coexist(self) -> None:
        """Pinned: domain layer does NOT enforce mutual exclusion between
        answer and error_info. qa_node sets a user-safe `answer` AND
        populates `error_info` on the failure path — both fields are
        present together in the failure case. The translator handles the
        AnswerGenerated-vs-ModelInvocationFailed dispatch.
        """
        info = ErrorInfo(exception_type="X", exception_message="Y")
        exchange = _make_exchange(
            answer="I couldn't reach the model.",
            error_info=info,
        )
        assert exchange.answer == "I couldn't reach the model."
        assert exchange.error_info == info


class TestQAExchangeReplaceRoundTrip:

    def test_replace_without_error_info(self) -> None:
        """V2 shape: replace flows context and answer additions through the
        exchange without touching error_info."""
        original = _make_exchange(question="What is LangGraph?")
        with_context = replace(original, context="Some context.")
        with_answer = replace(with_context, answer="An answer.")

        assert with_answer.question == "What is LangGraph?"
        assert with_answer.context == "Some context."
        assert with_answer.answer == "An answer."
        assert with_answer.error_info is None

    def test_replace_with_error_info(self) -> None:
        """V3a shape: replace flows error_info onto the exchange in the
        failure path, preserving question/context/answer alongside."""
        info = ErrorInfo(exception_type="X", exception_message="Y")
        original = _make_exchange(
            question="What is LangGraph?",
            context="Some context.",
        )
        failed = replace(
            original,
            answer="An error occurred.",
            error_info=info,
        )

        assert failed.question == "What is LangGraph?"
        assert failed.context == "Some context."
        assert failed.answer == "An error occurred."
        assert failed.error_info == info

    def test_replace_can_clear_error_info(self) -> None:
        info = ErrorInfo(exception_type="X", exception_message="Y")
        failed = _make_exchange(error_info=info)
        cleared = replace(failed, error_info=None)
        assert cleared.error_info is None


class TestQAExchangeFrozen:

    def test_is_frozen(self) -> None:
        exchange = _make_exchange()
        with pytest.raises(FrozenInstanceError):
            setattr(exchange, "answer", "different")

    def test_error_info_field_is_frozen(self) -> None:
        exchange = _make_exchange()
        with pytest.raises(FrozenInstanceError):
            setattr(exchange, "error_info", ErrorInfo("X", "Y"))