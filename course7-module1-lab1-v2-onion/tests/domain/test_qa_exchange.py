from dataclasses import FrozenInstanceError

import pytest

from domain.qa_exchange import QAExchange


def test_is_frozen_dataclass_cannot_be_mutated() -> None:
    exchange = QAExchange(question="What is LangGraph?")
    with pytest.raises(FrozenInstanceError):
        setattr(exchange, "question", "Something else")


def test_context_and_answer_default_to_none() -> None:
    exchange = QAExchange(question="What is LangGraph?")
    assert exchange.context is None
    assert exchange.answer is None


def test_all_fields_accessible() -> None:
    exchange = QAExchange(
        question="What is LangGraph?",
        context="LangGraph is a state machine framework.",
        answer="A state machine framework with LLM-friendly ergonomics.",
    )
    assert exchange.question == "What is LangGraph?"
    assert exchange.context == "LangGraph is a state machine framework."
    assert exchange.answer == "A state machine framework with LLM-friendly ergonomics."


def test_non_empty_question_satisfies_invariant() -> None:
    exchange = QAExchange(question="What?")
    assert exchange.question == "What?"


def test_whitespace_only_question_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        QAExchange(question="   \t\n  ")