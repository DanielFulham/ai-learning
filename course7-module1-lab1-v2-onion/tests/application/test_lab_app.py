from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from application.interfaces.auth_agent_service_interface import AuthAgentServiceInterface
from application.interfaces.counter_agent_service_interface import (
    CounterAgentServiceInterface,
)
from application.interfaces.qa_agent_service_interface import QAAgentServiceInterface
from application.lab_app import LabApp


def _make_lab_app() -> LabApp:
    return LabApp(
        auth=MagicMock(spec=AuthAgentServiceInterface),
        qa=MagicMock(spec=QAAgentServiceInterface),
        counter=MagicMock(spec=CounterAgentServiceInterface),
    )


def test_lab_app_exposes_three_services_by_name() -> None:
    app = _make_lab_app()
    assert hasattr(app, "auth")
    assert hasattr(app, "qa")
    assert hasattr(app, "counter")


def test_lab_app_is_frozen() -> None:
    app = _make_lab_app()
    with pytest.raises(FrozenInstanceError):
        setattr(app, "auth", None)