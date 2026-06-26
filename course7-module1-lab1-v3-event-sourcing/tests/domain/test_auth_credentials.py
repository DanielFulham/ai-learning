from dataclasses import FrozenInstanceError

import pytest

from domain.auth_credentials import AuthCredentials


class TestAuthCredentialsConstruction:

    def test_all_none_defaults(self) -> None:
        """Pinned: AuthCredentials starts empty — all fields default to None.
        The carrier fills in across the graph traversal."""
        creds = AuthCredentials()
        assert creds.username is None
        assert creds.password is None
        assert creds.is_authenticated is None
        assert creds.message is None

    def test_username_populated(self) -> None:
        creds = AuthCredentials(username="alice")
        assert creds.username == "alice"
        assert creds.password is None

    def test_password_populated(self) -> None:
        creds = AuthCredentials(password="secret")
        assert creds.password == "secret"
        assert creds.username is None

    def test_is_authenticated_true(self) -> None:
        creds = AuthCredentials(username="alice", is_authenticated=True)
        assert creds.is_authenticated is True

    def test_is_authenticated_false(self) -> None:
        creds = AuthCredentials(is_authenticated=False)
        assert creds.is_authenticated is False

    def test_message_populated(self) -> None:
        creds = AuthCredentials(message="Authentication successful! Welcome.")
        assert creds.message == "Authentication successful! Welcome."

    def test_all_fields_populated(self) -> None:
        creds = AuthCredentials(
            username="alice",
            password="secret",
            is_authenticated=True,
            message="Authentication successful! Welcome.",
        )
        assert creds.username == "alice"
        assert creds.password == "secret"
        assert creds.is_authenticated is True
        assert creds.message == "Authentication successful! Welcome."


class TestAuthCredentialsFrozen:

    def test_username_is_immutable(self) -> None:
        creds = AuthCredentials(username="alice")
        with pytest.raises(FrozenInstanceError):
            setattr(creds, "username", "bob")

    def test_password_is_immutable(self) -> None:
        creds = AuthCredentials(password="secret")
        with pytest.raises(FrozenInstanceError):
            setattr(creds, "password", "other")

    def test_is_authenticated_is_immutable(self) -> None:
        creds = AuthCredentials(is_authenticated=True)
        with pytest.raises(FrozenInstanceError):
            setattr(creds, "is_authenticated", False)

    def test_message_is_immutable(self) -> None:
        creds = AuthCredentials(message="hello")
        with pytest.raises(FrozenInstanceError):
            setattr(creds, "message", "different")


class TestAuthCredentialsNoPostInitValidation:

    def test_is_authenticated_true_with_no_username_is_allowed(self) -> None:
        """Pinned: no __post_init__ validation. V2's AuthCredentials
        enforced is_authenticated=True requires a non-empty username;
        V3b's carrier admits any combination — validation is the
        translator's job at the event boundary."""
        creds = AuthCredentials(is_authenticated=True, username=None)
        assert creds.is_authenticated is True
        assert creds.username is None
