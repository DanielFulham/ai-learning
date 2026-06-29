from dataclasses import fields
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID, uuid4

import pytest

from application.event_translation.auth_translator import (
    _LOGIN_FAILED_MESSAGE,
    _LOGIN_SUCCEEDED_MESSAGE,
    translate_auth_update,
)
from application.graph_builders import (
    AUTH_FAILURE_NODE_NAME,
    AUTH_INPUT_NODE_NAME,
    AUTH_SUCCESS_NODE_NAME,
    AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
)
from domain.auth_credentials import AuthCredentials
from domain.events.auth_events import LoginAttempted, LoginFailed, LoginSucceeded


_FIXED_TIME = datetime(2026, 6, 26, 12, 0, tzinfo=timezone.utc)


def _fixed_clock() -> Callable[[], datetime]:
    return lambda: _FIXED_TIME


def _counting_clock() -> tuple[Callable[[], datetime], dict[str, int]]:
    counter = {"calls": 0}

    def clock() -> datetime:
        counter["calls"] += 1
        return _FIXED_TIME

    return clock, counter


def _make_creds(
    username: str | None = "alice",
    password: str | None = "secret",
    is_authenticated: bool | None = None,
    message: str | None = None,
) -> AuthCredentials:
    return AuthCredentials(
        username=username,
        password=password,
        is_authenticated=is_authenticated,
        message=message,
    )


def _delta(credentials: AuthCredentials) -> dict[str, Any]:
    return {"credentials": credentials}


class TestTranslateInputNodeUpdate:

    def test_emits_exactly_one_login_attempted(self) -> None:
        run_id = uuid4()
        events = translate_auth_update(
            node_name=AUTH_INPUT_NODE_NAME,
            state_delta=_delta(_make_creds()),
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert len(events) == 1
        assert isinstance(events[0], LoginAttempted)

    def test_login_attempted_carries_username(self) -> None:
        run_id = uuid4()
        events = translate_auth_update(
            node_name=AUTH_INPUT_NODE_NAME,
            state_delta=_delta(_make_creds(username="bob")),
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert events[0].username == "bob"

    def test_login_attempted_carries_run_id_as_aggregate_id(self) -> None:
        run_id = uuid4()
        events = translate_auth_update(
            node_name=AUTH_INPUT_NODE_NAME,
            state_delta=_delta(_make_creds()),
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert events[0].aggregate_id == run_id

    def test_login_attempted_occurred_at_from_clock(self) -> None:
        run_id = uuid4()
        events = translate_auth_update(
            node_name=AUTH_INPUT_NODE_NAME,
            state_delta=_delta(_make_creds()),
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert events[0].occurred_at == _FIXED_TIME


class TestTranslateValidateCredentialsNodeUpdate:

    def test_is_authenticated_true_emits_login_succeeded(self) -> None:
        run_id = uuid4()
        creds = _make_creds(is_authenticated=True)
        events = translate_auth_update(
            node_name=AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
            state_delta=_delta(creds),
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert len(events) == 1
        assert isinstance(events[0], LoginSucceeded)

    def test_is_authenticated_false_emits_login_failed(self) -> None:
        run_id = uuid4()
        creds = _make_creds(is_authenticated=False)
        events = translate_auth_update(
            node_name=AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
            state_delta=_delta(creds),
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert len(events) == 1
        assert isinstance(events[0], LoginFailed)

    def test_is_authenticated_none_raises(self) -> None:
        """Pinned: is_authenticated=None at the verdict node is a node
        implementation bug — ValidateCredentialsNode must set a bool."""
        run_id = uuid4()
        creds = _make_creds(is_authenticated=None)
        with pytest.raises(ValueError, match="is_authenticated=None"):
            translate_auth_update(
                node_name=AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
                state_delta=_delta(creds),
                run_id=run_id,
                clock=_fixed_clock(),
            )

    def test_login_succeeded_carries_username(self) -> None:
        run_id = uuid4()
        creds = _make_creds(username="carol", is_authenticated=True)
        events = translate_auth_update(
            node_name=AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
            state_delta=_delta(creds),
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert events[0].username == "carol"

    def test_login_failed_carries_username(self) -> None:
        run_id = uuid4()
        creds = _make_creds(username="dave", is_authenticated=False)
        events = translate_auth_update(
            node_name=AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
            state_delta=_delta(creds),
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert events[0].username == "dave"


class TestTranslateSuccessAndFailureNodes:

    def test_success_node_returns_empty_list(self) -> None:
        """Pinned: SuccessNode is a UI-facing message stamp — no event."""
        run_id = uuid4()
        creds = _make_creds(
            is_authenticated=True, message="Authentication successful! Welcome."
        )
        events = translate_auth_update(
            node_name=AUTH_SUCCESS_NODE_NAME,
            state_delta=_delta(creds),
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert events == []

    def test_failure_node_returns_empty_list(self) -> None:
        """Pinned: FailureNode is a retry-state reset — no event."""
        run_id = uuid4()
        creds = _make_creds(
            username=None, password=None, is_authenticated=False, message="Failed."
        )
        events = translate_auth_update(
            node_name=AUTH_FAILURE_NODE_NAME,
            state_delta=_delta(creds),
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert events == []


class TestTranslateUnknownNodeRaises:

    def test_unknown_node_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown auth node"):
            translate_auth_update(
                node_name="GhostNode",
                state_delta=_delta(_make_creds()),
                run_id=uuid4(),
                clock=_fixed_clock(),
            )


class TestTwoStageNarrowing:

    def test_missing_credentials_key_raises(self) -> None:
        with pytest.raises(ValueError, match="missing or malformed"):
            translate_auth_update(
                node_name=AUTH_INPUT_NODE_NAME,
                state_delta={},
                run_id=uuid4(),
                clock=_fixed_clock(),
            )

    def test_wrong_type_in_credentials_raises(self) -> None:
        """Pinned: credentials of wrong type raises at the second narrowing
        stage, not silently passes. Same shape as QA translator's guard."""
        with pytest.raises(ValueError, match="missing or malformed"):
            translate_auth_update(
                node_name=AUTH_INPUT_NODE_NAME,
                state_delta={"credentials": "not-an-AuthCredentials"},
                run_id=uuid4(),
                clock=_fixed_clock(),
            )


class TestCodecHelperF03Policy:

    def test_login_attempted_has_no_password_field(self) -> None:
        """Pinned: C1 structural belt — password is absent from
        LoginAttempted's dataclass field set. This test mirrors the C1
        test; the translator test confirms the procedural guarantee: no
        construction path through translate_auth_update ever touches
        credentials.password."""
        field_names = {f.name for f in fields(LoginAttempted)}
        assert "password" not in field_names

    def test_codec_helper_raises_when_username_none_at_input_node(self) -> None:
        """Pinned: F03 procedural belt — the codec helper raises ValueError
        rather than silently constructing a LoginAttempted with username=None.
        LoginAttempted.username is str (non-nullable); the guard enforces
        the contract at construction time."""
        creds = _make_creds(username=None)
        with pytest.raises(ValueError, match="LoginAttempted requires credentials.username"):
            translate_auth_update(
                node_name=AUTH_INPUT_NODE_NAME,
                state_delta=_delta(creds),
                run_id=uuid4(),
                clock=_fixed_clock(),
            )

    def test_codec_helper_raises_when_username_none_at_succeeded(self) -> None:
        creds = _make_creds(username=None, is_authenticated=True)
        with pytest.raises(ValueError, match="LoginSucceeded requires credentials.username"):
            translate_auth_update(
                node_name=AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
                state_delta=_delta(creds),
                run_id=uuid4(),
                clock=_fixed_clock(),
            )

    def test_codec_helper_raises_when_username_none_at_failed(self) -> None:
        creds = _make_creds(username=None, is_authenticated=False)
        with pytest.raises(ValueError, match="LoginFailed requires credentials.username"):
            translate_auth_update(
                node_name=AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
                state_delta=_delta(creds),
                run_id=uuid4(),
                clock=_fixed_clock(),
            )


class TestMessageHandlingDecoupling:

    def test_login_succeeded_message_is_hardcoded_constant(self) -> None:
        """Pinned: event-log message is decoupled from UI message.
        The translator uses its hardcoded _LOGIN_SUCCEEDED_MESSAGE constant,
        not credentials.message. Even when credentials carries a different
        message (e.g. from success_node's UI stamp), the event message
        remains the audit-log wording."""
        creds = _make_creds(
            is_authenticated=True,
            message="Authentication successful! Welcome.",  # UI message from success_node
        )
        events = translate_auth_update(
            node_name=AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
            state_delta=_delta(creds),
            run_id=uuid4(),
            clock=_fixed_clock(),
        )

        assert len(events) == 1
        assert isinstance(events[0], LoginSucceeded)
        assert events[0].message == _LOGIN_SUCCEEDED_MESSAGE

    def test_login_failed_message_is_hardcoded_constant(self) -> None:
        """Pinned: same decoupling for the failure event."""
        creds = _make_creds(
            is_authenticated=False,
            message="Not Successful, please try again!",  # UI message from failure_node
        )
        events = translate_auth_update(
            node_name=AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
            state_delta=_delta(creds),
            run_id=uuid4(),
            clock=_fixed_clock(),
        )

        assert len(events) == 1
        assert isinstance(events[0], LoginFailed)
        assert events[0].message == _LOGIN_FAILED_MESSAGE

    def test_ui_message_can_differ_from_event_message(self) -> None:
        """Pinned: the two messages may diverge without coupling failure.
        Pass a credentials object with credentials.message='something else'
        and assert the event still carries the constant."""
        creds = _make_creds(is_authenticated=True, message="something entirely different")
        events = translate_auth_update(
            node_name=AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
            state_delta=_delta(creds),
            run_id=uuid4(),
            clock=_fixed_clock(),
        )

        assert isinstance(events[0], LoginSucceeded)
        assert events[0].message == _LOGIN_SUCCEEDED_MESSAGE
        assert events[0].message != "something entirely different"


class TestTranslatorAsCanonicalSource:

    def test_event_ids_are_unique_across_calls(self) -> None:
        """Pinned: event_id is uuid4 at translation time. Two calls with
        identical inputs produce events with different event_ids."""
        creds = _make_creds()
        run_id = uuid4()
        delta = _delta(creds)

        first = translate_auth_update(AUTH_INPUT_NODE_NAME, delta, run_id, _fixed_clock())
        second = translate_auth_update(AUTH_INPUT_NODE_NAME, delta, run_id, _fixed_clock())

        assert first[0].event_id != second[0].event_id

    def test_occurred_at_comes_from_clock(self) -> None:
        """Pinned: occurred_at is the clock()'s return value, not datetime.now()."""
        run_id = uuid4()
        events = translate_auth_update(
            node_name=AUTH_INPUT_NODE_NAME,
            state_delta=_delta(_make_creds()),
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert events[0].occurred_at == _FIXED_TIME

    def test_aggregate_id_is_run_id(self) -> None:
        run_id = uuid4()
        events = translate_auth_update(
            node_name=AUTH_INPUT_NODE_NAME,
            state_delta=_delta(_make_creds()),
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert events[0].aggregate_id == run_id

    def test_clock_called_exactly_once_per_emitted_event(self) -> None:
        """Pinned: one clock call per event emitted."""
        clock, counter = _counting_clock()
        translate_auth_update(
            node_name=AUTH_INPUT_NODE_NAME,
            state_delta=_delta(_make_creds()),
            run_id=uuid4(),
            clock=clock,
        )

        assert counter["calls"] == 1

    def test_schema_version_defaults_to_1(self) -> None:
        """Pinned: schema_version=1 across all Auth events at V3b."""
        run_id = uuid4()
        events = translate_auth_update(
            node_name=AUTH_INPUT_NODE_NAME,
            state_delta=_delta(_make_creds()),
            run_id=run_id,
            clock=_fixed_clock(),
        )

        assert events[0].schema_version == 1
