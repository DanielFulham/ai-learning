from typing import NamedTuple
from unittest.mock import MagicMock

import pytest

from application.interfaces.auth_agent_service_interface import AuthAgentServiceInterface
from application.interfaces.counter_agent_service_interface import (
    CounterAgentServiceInterface,
)
from application.interfaces.qa_agent_service_interface import QAAgentServiceInterface
from application.lab_app import LabApp
from demo import main, run_all, run_auth, run_counter, run_qa
from domain.auth_credentials import AuthCredentials
from domain.counter_tick import CounterTick
from domain.qa_exchange import QAExchange


class LabAppAndMocks(NamedTuple):
    app: LabApp
    auth_mock: MagicMock
    qa_mock: MagicMock
    counter_mock: MagicMock


def _make_lab_app_and_mocks() -> LabAppAndMocks:
    auth_mock = MagicMock(spec=AuthAgentServiceInterface)
    auth_mock.run.return_value = AuthCredentials(
        username="test_user",
        is_authenticated=True,
        message="Authentication successful! Welcome.",
    )

    qa_mock = MagicMock(spec=QAAgentServiceInterface)
    qa_mock.run.return_value = QAExchange(
        question="What is LangGraph?",
        context="some context",
        answer="A state machine framework.",
    )

    counter_mock = MagicMock(spec=CounterAgentServiceInterface)
    counter_mock.run.return_value = CounterTick(n=13, letter="a")

    app = LabApp(auth=auth_mock, qa=qa_mock, counter=counter_mock)
    return LabAppAndMocks(app=app, auth_mock=auth_mock, qa_mock=qa_mock, counter_mock=counter_mock)


def test_run_auth_invokes_auth_service_once(capsys) -> None:
    bundle = _make_lab_app_and_mocks()
    run_auth(bundle.app)
    bundle.auth_mock.run.assert_called_once()
    bundle.qa_mock.run.assert_not_called()
    bundle.counter_mock.run.assert_not_called()


def test_run_qa_invokes_qa_service_three_times(capsys) -> None:
    """The three canned demo questions — pinned because changing the set
    is a deliberate decision (the third is the hallucination case)."""
    bundle = _make_lab_app_and_mocks()
    run_qa(bundle.app)
    assert bundle.qa_mock.run.call_count == 3


def test_run_qa_third_question_is_hallucination_case() -> None:
    """V1's eval-failure case preserved as the third demo question."""
    bundle = _make_lab_app_and_mocks()
    run_qa(bundle.app)
    third_question = bundle.qa_mock.run.call_args_list[2][0][0]
    assert "guided project" in third_question


def test_run_counter_invokes_counter_once(capsys) -> None:
    bundle = _make_lab_app_and_mocks()
    run_counter(bundle.app)
    bundle.counter_mock.run.assert_called_once()


def test_run_all_invokes_all_three_services_regardless_of_auth_verdict(capsys) -> None:
    """Even if Auth fails, Counter and QA still run. Coupling lives at the
    entry point, but the demo deliberately does not gate the others on
    the auth verdict — integrated UX without graph-level coupling."""
    bundle = _make_lab_app_and_mocks()
    bundle.auth_mock.run.return_value = AuthCredentials(
        username="test_user",
        is_authenticated=False,
        message="Not Successful, please try again!",
    )
    run_all(bundle.app)
    bundle.auth_mock.run.assert_called_once()
    assert bundle.qa_mock.run.call_count == 3
    bundle.counter_mock.run.assert_called_once()


def test_main_with_unknown_workflow_exits_non_zero(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["bogus"])
    assert excinfo.value.code != 0


def test_main_with_missing_workflow_exits_non_zero(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0


def test_main_with_counter_dispatches_to_counter(monkeypatch, capsys) -> None:
    """End-to-end dispatch test — `main(['counter'])` builds the container
    and calls run_counter. Mocks the container to avoid the Ollama
    instantiation path."""
    bundle = _make_lab_app_and_mocks()
    monkeypatch.setattr("demo.initialise", lambda **kwargs: bundle.app)

    exit_code = main(["counter"])

    assert exit_code == 0
    bundle.counter_mock.run.assert_called_once()
    bundle.auth_mock.run.assert_not_called()
    bundle.qa_mock.run.assert_not_called()


def test_main_with_provider_openai_passes_use_openai_true(monkeypatch) -> None:
    """The --provider openai flag becomes use_openai=True on initialise."""
    bundle = _make_lab_app_and_mocks()
    captured_kwargs: dict = {}

    def fake_initialise(**kwargs):
        captured_kwargs.update(kwargs)
        return bundle.app

    monkeypatch.setattr("demo.initialise", fake_initialise)

    main(["counter", "--provider", "openai"])

    assert captured_kwargs["use_openai"] is True


def test_main_default_provider_passes_use_openai_false(monkeypatch) -> None:
    """When --provider is not specified, default is ollama → use_openai=False."""
    bundle = _make_lab_app_and_mocks()
    captured_kwargs: dict = {}

    def fake_initialise(**kwargs):
        captured_kwargs.update(kwargs)
        return bundle.app

    monkeypatch.setattr("demo.initialise", fake_initialise)

    main(["counter"])

    assert captured_kwargs["use_openai"] is False


def test_main_all_workflow_passes_use_scripted_auth_input_true(monkeypatch) -> None:
    """`demo.py all` triggers use_scripted_auth_input=True so the auth
    section doesn't block on stdin during the integrated demo."""
    bundle = _make_lab_app_and_mocks()
    captured_kwargs: dict = {}

    def fake_initialise(**kwargs):
        captured_kwargs.update(kwargs)
        return bundle.app

    monkeypatch.setattr("demo.initialise", fake_initialise)

    main(["all"])

    assert captured_kwargs["use_scripted_auth_input"] is True


def test_main_single_workflow_passes_use_scripted_auth_input_false(monkeypatch) -> None:
    """`demo.py auth` runs interactively — use_scripted_auth_input=False."""
    bundle = _make_lab_app_and_mocks()
    captured_kwargs: dict = {}

    def fake_initialise(**kwargs):
        captured_kwargs.update(kwargs)
        return bundle.app

    monkeypatch.setattr("demo.initialise", fake_initialise)

    main(["auth"])

    assert captured_kwargs["use_scripted_auth_input"] is False