from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest

from domain.events.base import BaseAgentEvent


def _make_base_event(
    event_id: UUID | None = None,
    aggregate_id: UUID | None = None,
    occurred_at: datetime | None = None,
    schema_version: int = 1,
) -> BaseAgentEvent:
    return BaseAgentEvent(
        event_id=event_id or uuid4(),
        aggregate_id=aggregate_id or uuid4(),
        occurred_at=occurred_at or datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc),
        schema_version=schema_version,
    )


def _full_kwargs() -> dict[str, Any]:
    return {
        "event_id": uuid4(),
        "aggregate_id": uuid4(),
        "occurred_at": datetime(2026, 6, 23, tzinfo=timezone.utc),
    }


class TestBaseAgentEvent:

    def test_all_required_fields_present(self) -> None:
        """Pinned: the four common-event fields are the architectural contract.
        Removing any of them is a breaking change to every consumer of the
        event store."""
        field_names = {f.name for f in fields(BaseAgentEvent)}
        assert field_names == {
            "event_id",
            "aggregate_id",
            "occurred_at",
            "schema_version",
        }

    def test_schema_version_defaults_to_1(self) -> None:
        """Pinned: V3a is schema_version=1 across the board. Bumping the
        default is a deliberate cross-cutting decision; new event types
        or fields land as additive shapes without bumping this default
        unless the existing event's structure changes."""
        event = _make_base_event()
        assert event.schema_version == 1

    @pytest.mark.parametrize(
        "missing_key", ["event_id", "aggregate_id", "occurred_at"]
    )
    def test_identity_fields_have_no_defaults(self, missing_key: str) -> None:
        """Pinned: event_id, aggregate_id, occurred_at must be passed explicitly.
        The translator is the canonical source of these values; defaulting
        them would let a node or service silently construct events with the
        wrong provenance.

        Verified by attempting construction without each in turn. The kwargs
        splat keeps pyright honest about the production-code contract while
        letting the runtime TypeError pin the design decision.
        """
        kwargs = _full_kwargs()
        del kwargs[missing_key]
        with pytest.raises(TypeError):
            BaseAgentEvent(**kwargs)

    def test_is_frozen(self) -> None:
        event = _make_base_event()
        with pytest.raises(FrozenInstanceError):
            setattr(event, "schema_version", 99)

    def test_kw_only_construction(self) -> None:
        """Pinned: kw_only=True is the load-bearing decorator choice.
        Positional construction must fail — this guarantees the inheritance
        ordering (subclass adds non-default fields after base's defaulted
        schema_version) stays legal forever.

        The class is bound through an Any-typed name so pyright doesn't
        statically flag the positional call. The runtime TypeError is what
        pins the design decision; pyright's own check on production-code
        call sites is a separate enforcement layer.
        """
        cls: Any = BaseAgentEvent
        args = (
            uuid4(),
            uuid4(),
            datetime(2026, 6, 23, tzinfo=timezone.utc),
        )
        with pytest.raises(TypeError):
            cls(*args)