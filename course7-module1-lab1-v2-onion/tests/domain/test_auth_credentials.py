from dataclasses import FrozenInstanceError

import pytest

from domain.auth_credentials import AuthCredentials


def test_is_frozen_dataclass_cannot_be_mutated() -> None:
    credentials = AuthCredentials(username="test_user", password="secret")
    with pytest.raises(FrozenInstanceError):
        setattr(credentials, "username", "someone_else")


def test_defaults_are_all_none() -> None:
    credentials = AuthCredentials()
    assert credentials.username is None
    assert credentials.password is None
    assert credentials.is_authenticated is None
    assert credentials.message is None


def test_all_fields_accessible() -> None:
    credentials = AuthCredentials(
        username="test_user",
        password="secure_password",
        is_authenticated=True,
        message="Authentication successful! Welcome.",
    )
    assert credentials.username == "test_user"
    assert credentials.password == "secure_password"
    assert credentials.is_authenticated is True
    assert credentials.message == "Authentication successful! Welcome."


def test_authenticated_true_with_valid_username_succeeds() -> None:
    credentials = AuthCredentials(username="test_user", is_authenticated=True)
    assert credentials.is_authenticated is True
    assert credentials.username == "test_user"


def test_authenticated_true_with_none_username_raises() -> None:
    with pytest.raises(ValueError, match="non-empty username"):
        AuthCredentials(is_authenticated=True)


def test_authenticated_true_with_empty_username_raises() -> None:
    with pytest.raises(ValueError, match="non-empty username"):
        AuthCredentials(username="", is_authenticated=True)