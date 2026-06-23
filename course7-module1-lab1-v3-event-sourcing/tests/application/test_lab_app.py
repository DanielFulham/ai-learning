from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from application.interfaces.qa_agent_service_interface import (
    QAAgentServiceInterface,
)
from application.lab_app import LabApp


class TestLabApp:

    def test_qa_field_accessible(self) -> None:
        qa = MagicMock(spec=QAAgentServiceInterface)
        app = LabApp(qa=qa)
        assert app.qa is qa

    def test_is_frozen(self) -> None:
        """Pinned: LabApp is immutable. Rewiring requires calling
        initialise() again — no field reassignment after construction."""
        app = LabApp(qa=MagicMock(spec=QAAgentServiceInterface))
        with pytest.raises(FrozenInstanceError):
            setattr(app, "qa", MagicMock(spec=QAAgentServiceInterface))