from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone
from typing import Any, assert_never
from uuid import UUID, uuid4

import pytest

from domain.events.base import BaseAgentEvent
from domain.events.auth_events import (
    AuthEvent,
    LoginAttempted,
    LoginFailed,
    LoginSucceeded,
)


def _common_kwargs(
    event_id: UUID | None = None,
    aggregate_id: UUID | None = None,
    occurred_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "event_id": event_id or uuid4(),
        "aggregate_id": aggregate_id or uuid4(),
        "occurred_at": occurred_at or datetime(2026, 6, 26, 12, 0, tzinfo=timezone.utc),
    }


def _make_login_attempted(username: str = "test_user") -> LoginAttempted:
    return LoginAttempted(**_common_kwargs(), username=username)


def _make_login_succeeded(
    username: str = "test_user",
    message: str = "Authentication successful! Welcome.",
) -> LoginSucceeded:
    return LoginSucceeded(**_common_kwargs(), username=username, message=message)


def _make_login_failed(
    username: str = "test_user",
    message: str = "Not Successful, please try again!",
) -> LoginFailed:
    return LoginFailed(**_common_kwargs(), username=username, message=message)


class TestLoginAttempted:

    def test_inherits_base_event(self) -> None:
        """Pinned: BaseAgentEvent inheritance is structural — event_id,
        aggregate_id, occurred_at, schema_version must be reachable
        on every event type the translator emits."""
        event = _make_login_attempted()
        assert isinstance(event, BaseAgentEvent)

    def test_carries_username_payload(self) -> None:
        event = _make_login_attempted(username="alice")
        assert event.username == "alice"

    def test_is_frozen(self) -> None:
        event = _make_login_attempted()
        with pytest.raises(FrozenInstanceError):
            setattr(event, "username", "different")

    def test_rejects_construction_without_no_default_fields(self) -> None:
        """Pinned: event_id, aggregate_id, and occurred_at carry no defaults.
        Omitting them must raise TypeError — the translator is the canonical
        source of identity fields, not the dataclass default mechanism."""
        with pytest.raises(TypeError):
            kwargs: dict[str, Any] = {"username": "test_user"}
            LoginAttempted(**kwargs)

    def test_password_absent_from_field_set(self) -> None:
        """Pinned: password is absent from LoginAttempted's field set by
        structural F03 policy. The persisted payload can never carry a
        password field because it is not in the dataclass. This is the C1
        half of the belt-and-braces guarantee; C3's translator codec helper
        is the procedural half."""
        field_names = {f.name for f in fields(LoginAttempted)}
        assert "password" not in field_names


class TestLoginSucceeded:

    def test_inherits_base_event(self) -> None:
        event = _make_login_succeeded()
        assert isinstance(event, BaseAgentEvent)

    def test_carries_username_and_message(self) -> None:
        event = _make_login_succeeded(
            username="alice", message="Authentication successful! Welcome."
        )
        assert event.username == "alice"
        assert event.message == "Authentication successful! Welcome."

    def test_is_frozen(self) -> None:
        event = _make_login_succeeded()
        with pytest.raises(FrozenInstanceError):
            setattr(event, "username", "different")

    def test_rejects_construction_without_no_default_fields(self) -> None:
        """Pinned: event_id, aggregate_id, and occurred_at carry no defaults."""
        with pytest.raises(TypeError):
            kwargs: dict[str, Any] = {"username": "test_user", "message": "Welcome."}
            LoginSucceeded(**kwargs)


class TestLoginFailed:

    def test_inherits_base_event(self) -> None:
        event = _make_login_failed()
        assert isinstance(event, BaseAgentEvent)

    def test_carries_username_and_message(self) -> None:
        event = _make_login_failed(
            username="alice", message="Not Successful, please try again!"
        )
        assert event.username == "alice"
        assert event.message == "Not Successful, please try again!"

    def test_is_frozen(self) -> None:
        event = _make_login_failed()
        with pytest.raises(FrozenInstanceError):
            setattr(event, "username", "different")

    def test_rejects_construction_without_no_default_fields(self) -> None:
        """Pinned: event_id, aggregate_id, and occurred_at carry no defaults."""
        with pytest.raises(TypeError):
            kwargs: dict[str, Any] = {"username": "test_user", "message": "Failed."}
            LoginFailed(**kwargs)


class TestAuthEventUnion:

    def test_exhaustive_match_covers_every_member(self) -> None:
        """Pinned: AuthEvent is closed within the Auth service. The translator
        boundary uses match-case + typing.assert_never to enforce
        exhaustiveness at type-check time. This test instantiates one of
        each variant and routes through a placeholder match — if a new
        member is added to AuthEvent without updating this match, pyright
        flags assert_never as reachable with an unhandled type.

        The placeholder handler is intentionally minimal — the actual
        translator-output dispatch lands in the Auth translator commit.
        This test pins the union shape, not the dispatch behaviour.
        """

        def handle(event: AuthEvent) -> str:
            match event:
                case LoginAttempted():
                    return "login_attempted"
                case LoginSucceeded():
                    return "login_succeeded"
                case LoginFailed():
                    return "login_failed"
                case _:
                    assert_never(event)

        assert handle(_make_login_attempted()) == "login_attempted"
        assert handle(_make_login_succeeded()) == "login_succeeded"
        assert handle(_make_login_failed()) == "login_failed"
