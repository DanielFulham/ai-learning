from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from application.qa_nodes import context_provider_node, make_qa_node
from domain.qa_exchange import QAExchange
from domain.state_schemas import QAState


def _make_model_returning(text: str) -> MagicMock:
    """Mock BaseChatModel with .invoke returning an AIMessage carrying
    the given str content. Real AIMessage so the str narrowing in
    invoke_text passes through naturally."""
    model = MagicMock(spec=BaseChatModel)
    model.invoke.return_value = AIMessage(content=text)
    return model


def _make_model_raising(exc: BaseException) -> MagicMock:
    model = MagicMock(spec=BaseChatModel)
    model.invoke.side_effect = exc
    return model


def _exchange_from(result: QAState) -> QAExchange:
    """Narrow the TypedDict (total=False) access. QAState.exchange is
    optional in the type system but always present in any return from
    these nodes — this helper asserts the contract and returns the
    narrowed QAExchange."""
    exchange = result.get("exchange")
    assert exchange is not None, "node result must contain exchange"
    return exchange


def _make_state(exchange: QAExchange) -> QAState:
    state: QAState = {"exchange": exchange}
    return state


class TestContextProviderNodeKeywordMatch:

    def test_langgraph_keyword_emits_explicit_context_delta(self) -> None:
        result = context_provider_node(
            _make_state(QAExchange(question="What is LangGraph?"))
        )
        exchange = _exchange_from(result)
        assert exchange.context is not None
        assert "LangGraph" in exchange.context

    def test_guided_project_keyword_emits_explicit_context_delta(self) -> None:
        """The V2 hallucination is preserved as behaviour — 'guided project'
        in the question returns LangGraph context regardless of subject."""
        result = context_provider_node(
            _make_state(QAExchange(question="What is the best guided project?"))
        )
        exchange = _exchange_from(result)
        assert exchange.context is not None
        assert "LangGraph" in exchange.context


class TestContextProviderNodeObservabilityConsistencyLift:

    def test_no_match_returns_explicit_context_none_delta(self) -> None:
        """Pinned: V3a behaviour change. V2 returned `state` unchanged on a
        no-match; V3a returns an explicit `{"exchange": replace(current,
        context=None)}` delta. Guarantees the translator always sees a
        ContextRetrieved emission for every ContextNode execution.
        """
        result = context_provider_node(
            _make_state(QAExchange(question="What is the weather?"))
        )
        assert "exchange" in result
        exchange = _exchange_from(result)
        assert exchange.context is None

    def test_no_match_preserves_original_question(self) -> None:
        result = context_provider_node(
            _make_state(QAExchange(question="What is the weather?"))
        )
        exchange = _exchange_from(result)
        assert exchange.question == "What is the weather?"


class TestContextProviderNodeGuards:

    def test_missing_exchange_raises(self) -> None:
        empty: QAState = {}
        with pytest.raises(ValueError, match="requires an exchange"):
            context_provider_node(empty)


class TestQANodeNoContext:

    def test_returns_no_context_user_message(self) -> None:
        """Pinned: no-context is a successful answer path, not a failure.
        AnswerGenerated will be emitted (no error_info)."""
        node = make_qa_node(_make_model_returning("unused"))
        result = node(_make_state(QAExchange(question="Q", context=None)))

        exchange = _exchange_from(result)
        assert exchange.answer == (
            "I don't have enough context to answer your question."
        )
        assert exchange.error_info is None


class TestQANodeSuccess:

    def test_returns_answer_from_model(self) -> None:
        node = make_qa_node(_make_model_returning("The answer is 42."))
        result = node(_make_state(QAExchange(question="Q", context="C")))

        exchange = _exchange_from(result)
        assert exchange.answer == "The answer is 42."
        assert exchange.error_info is None


class TestQANodeRecoverableErrors:

    def test_httpx_error_populates_error_info_and_user_safe_answer(self) -> None:
        """Pinned: V3a behaviour change. Recoverable LLM errors populate
        `error_info` on the exchange AND set a user-safe `answer`. The
        translator's failure branch fires ModelInvocationFailed; the
        user-safe answer is for the UI only.
        """
        exc = httpx.ConnectError("Connection refused")
        node = make_qa_node(_make_model_raising(exc))
        result = node(_make_state(QAExchange(question="Q", context="C")))

        exchange = _exchange_from(result)
        assert exchange.error_info is not None
        assert exchange.error_info.exception_type == "ConnectError"
        assert "Connection refused" in exchange.error_info.exception_message
        assert exchange.answer == (
            "I couldn't reach the model right now. Please try again."
        )

    def test_connection_error_populates_error_info(self) -> None:
        exc = ConnectionError("network down")
        node = make_qa_node(_make_model_raising(exc))
        result = node(_make_state(QAExchange(question="Q", context="C")))

        exchange = _exchange_from(result)
        assert exchange.error_info is not None
        assert exchange.error_info.exception_type == "ConnectionError"
        assert exchange.error_info.exception_message == "network down"

    def test_timeout_error_populates_error_info(self) -> None:
        exc = TimeoutError("model took too long")
        node = make_qa_node(_make_model_raising(exc))
        result = node(_make_state(QAExchange(question="Q", context="C")))

        exchange = _exchange_from(result)
        assert exchange.error_info is not None
        assert exchange.error_info.exception_type == "TimeoutError"


class TestQANodeLogicBugsPropagate:

    def test_value_error_propagates_unchanged(self) -> None:
        """Pinned: logic bugs (ValueError, TypeError, KeyError,
        AttributeError) are NOT recoverable and propagate. The narrow
        except is the discipline boundary — application bugs surface as
        crashes, not as silently logged failure events."""
        exc = ValueError("programming error")
        node = make_qa_node(_make_model_raising(exc))

        with pytest.raises(ValueError, match="programming error"):
            node(_make_state(QAExchange(question="Q", context="C")))

    def test_type_error_propagates_unchanged(self) -> None:
        exc = TypeError("type bug")
        node = make_qa_node(_make_model_raising(exc))

        with pytest.raises(TypeError):
            node(_make_state(QAExchange(question="Q", context="C")))


class TestQANodeGuards:

    def test_missing_exchange_raises(self) -> None:
        node = make_qa_node(_make_model_returning("unused"))
        empty: QAState = {}
        with pytest.raises(ValueError, match="requires an exchange"):
            node(empty)