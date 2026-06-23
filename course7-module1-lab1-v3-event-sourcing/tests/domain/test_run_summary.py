from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from domain.error_info import ErrorInfo
from domain.run_summary import RunStatus, RunSummary


def _make_summary(
    answer: str | None = "An answer",
    error_info: ErrorInfo | None = None,
    final_status: RunStatus = "success",
) -> RunSummary:
    return RunSummary(
        run_id=uuid4(),
        started_at=datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 6, 23, 12, 0, 5, tzinfo=timezone.utc),
        question="Q",
        final_status=final_status,
        answer=answer,
        error_info=error_info,
        event_count=3,
    )


class TestRunSummary:

    def test_fields_accessible(self) -> None:
        run_id = uuid4()
        info = ErrorInfo(exception_type="X", exception_message="Y")
        started = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)
        completed = datetime(2026, 6, 23, 12, 0, 5, tzinfo=timezone.utc)

        summary = RunSummary(
            run_id=run_id,
            started_at=started,
            completed_at=completed,
            question="Q",
            final_status="failed",
            answer=None,
            error_info=info,
            event_count=4,
        )

        assert summary.run_id == run_id
        assert summary.started_at == started
        assert summary.completed_at == completed
        assert summary.question == "Q"
        assert summary.final_status == "failed"
        assert summary.answer is None
        assert summary.error_info == info
        assert summary.event_count == 4

    def test_is_frozen(self) -> None:
        summary = _make_summary()
        with pytest.raises(FrozenInstanceError):
            setattr(summary, "event_count", 99)

    def test_structural_equality(self) -> None:
        run_id = uuid4()
        started = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)
        a = RunSummary(
            run_id=run_id,
            started_at=started,
            completed_at=None,
            question="Q",
            final_status="incomplete",
            answer=None,
            error_info=None,
            event_count=1,
        )
        b = RunSummary(
            run_id=run_id,
            started_at=started,
            completed_at=None,
            question="Q",
            final_status="incomplete",
            answer=None,
            error_info=None,
            event_count=1,
        )
        assert a == b