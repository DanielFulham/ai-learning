from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from domain.qa_exchange import QAExchange
from domain.run_result import RunResult


class TestRunResult:

    def test_fields_accessible(self) -> None:
        run_id = uuid4()
        exchange = QAExchange(question="Q", answer="A")
        result = RunResult(exchange=exchange, run_id=run_id)
        assert result.exchange == exchange
        assert result.run_id == run_id

    def test_is_frozen(self) -> None:
        result = RunResult(
            exchange=QAExchange(question="Q"), run_id=uuid4()
        )
        with pytest.raises(FrozenInstanceError):
            setattr(result, "run_id", uuid4())

    def test_structural_equality(self) -> None:
        """Pinned: equality compares both fields. Two RunResults with the
        same exchange and same run_id compare equal — lets tests assert
        on result contents without holding references."""
        run_id = uuid4()
        exchange = QAExchange(question="Q", answer="A")
        a = RunResult(exchange=exchange, run_id=run_id)
        b = RunResult(exchange=exchange, run_id=run_id)
        assert a == b

    def test_different_run_id_not_equal(self) -> None:
        exchange = QAExchange(question="Q", answer="A")
        a = RunResult(exchange=exchange, run_id=uuid4())
        b = RunResult(exchange=exchange, run_id=uuid4())
        assert a != b