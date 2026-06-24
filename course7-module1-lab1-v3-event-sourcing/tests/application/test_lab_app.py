from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from application.interfaces.qa_agent_service_interface import (
    QAAgentServiceInterface,
)
from application.lab_app import LabApp
from interfaces.agent_event_store_interface import AgentEventStoreInterface


def _make_app() -> LabApp:
    return LabApp(
        qa=MagicMock(spec=QAAgentServiceInterface),
        event_store=MagicMock(spec=AgentEventStoreInterface),
    )


class TestLabApp:

    def test_qa_field_accessible(self) -> None:
        qa = MagicMock(spec=QAAgentServiceInterface)
        store = MagicMock(spec=AgentEventStoreInterface)
        app = LabApp(qa=qa, event_store=store)
        assert app.qa is qa

    def test_event_store_field_accessible(self) -> None:
        """Pinned: event_store is exposed on the bundle for read-side
        consumers (demo's RunSummaryProjection call, V3c's
        ThreadHistoryProjection). Same instance the services hold —
        the singleton contract surfaces as a shared reference."""
        qa = MagicMock(spec=QAAgentServiceInterface)
        store = MagicMock(spec=AgentEventStoreInterface)
        app = LabApp(qa=qa, event_store=store)
        assert app.event_store is store

    def test_is_frozen(self) -> None:
        app = _make_app()
        with pytest.raises(FrozenInstanceError):
            setattr(app, "qa", MagicMock(spec=QAAgentServiceInterface))

    def test_event_store_field_is_frozen(self) -> None:
        app = _make_app()
        with pytest.raises(FrozenInstanceError):
            setattr(app, "event_store", MagicMock(spec=AgentEventStoreInterface))