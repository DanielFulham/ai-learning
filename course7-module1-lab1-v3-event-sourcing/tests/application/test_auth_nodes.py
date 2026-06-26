from dataclasses import replace
from unittest.mock import MagicMock, call

import pytest

from application.auth_nodes import (
    _FAILURE_MESSAGE,
    _SUCCESS_MESSAGE,
    _VALID_PASSWORD,
    _VALID_USERNAME,
    failure_node,
    make_input_node,
    success_node,
    validate_credentials_node,
)
from domain.auth_credentials import AuthCredentials
from domain.state_schemas import AuthState
from interfaces.input_provider_interface import InputProviderInterface


def _state_with(credentials: AuthCredentials) -> AuthState:
    return {"credentials": credentials}


def _empty_creds() -> AuthCredentials:
    return AuthCredentials()


def _creds_from(result: AuthState) -> AuthCredentials:
    """Narrow the TypedDict (total=False) access. AuthState.credentials is
    optional in the type system but always present in any return from
    these nodes — this helper asserts the contract and returns the
    narrowed AuthCredentials."""
    creds = result.get("credentials")
    assert creds is not None, "node result must contain credentials"
    return creds

def _mock_provider(responses: list[str]) -> MagicMock:
    """Mock InputProviderInterface returning scripted responses."""
    provider = MagicMock(spec=InputProviderInterface)
    provider.prompt.side_effect = responses
    return provider


class TestMakeInputNode:

    def test_prompts_for_username_and_password_when_username_none(self) -> None:
        """Pinned: when credentials.username is None, input_node calls
        prompt twice — once for username, once for password."""
        provider = _mock_provider(["alice", "secret"])
        input_node = make_input_node(provider)

        result = input_node(_state_with(AuthCredentials()))

        assert provider.prompt.call_count == 2
        assert provider.prompt.call_args_list == [
            call("What is your username? "),
            call("Enter your password: "),
        ]
        assert result == {"credentials": AuthCredentials(username="alice", password="secret")}

    def test_reuses_existing_username_if_set(self) -> None:
        """Pinned: if credentials.username is already non-empty, the
        username prompt is skipped. Only password is prompted."""
        provider = _mock_provider(["secret"])
        input_node = make_input_node(provider)

        result = input_node(_state_with(AuthCredentials(username="alice")))

        assert provider.prompt.call_count == 1
        assert provider.prompt.call_args_list == [call("Enter your password: ")]
        assert result == {"credentials": AuthCredentials(username="alice", password="secret")}

    def test_returns_explicit_delta_with_replace(self) -> None:
        """Pinned: V3b observability-consistency lift — replace() preserves
        other carrier fields, returns explicit delta."""
        provider = _mock_provider(["bob", "pw"])
        input_node = make_input_node(provider)
        initial = AuthCredentials(is_authenticated=False, message="prior message")

        result = input_node(_state_with(initial))

        # replace() preserves is_authenticated and message from the carrier
        creds = _creds_from(result)
        assert creds.username == "bob"
        assert creds.password == "pw"

    def test_raises_when_credentials_missing_from_state(self) -> None:
        """Pinned: all nodes raise ValueError if credentials absent.
        The auth service pre-populates with AuthCredentials() before streaming."""
        provider = _mock_provider([])
        input_node = make_input_node(provider)

        with pytest.raises(ValueError, match="input_node requires credentials in state"):
            input_node({})

    def test_empty_string_username_triggers_prompt(self) -> None:
        """Pinned: empty string is treated as absent — prompt fires."""
        provider = _mock_provider(["alice", "secret"])
        input_node = make_input_node(provider)

        input_node(_state_with(AuthCredentials(username="")))

        assert provider.prompt.call_count == 2


class TestValidateCredentialsNode:

    def test_correct_credentials_set_is_authenticated_true(self) -> None:
        """Pinned: the exact V2 test pair (test_user / secure_password)
        produces is_authenticated=True."""
        creds = AuthCredentials(username=_VALID_USERNAME, password=_VALID_PASSWORD)
        result = validate_credentials_node(_state_with(creds))

        assert _creds_from(result).is_authenticated is True

    def test_wrong_password_sets_is_authenticated_false(self) -> None:
        creds = AuthCredentials(username=_VALID_USERNAME, password="wrong")
        result = validate_credentials_node(_state_with(creds))

        assert _creds_from(result).is_authenticated is False

    def test_wrong_username_sets_is_authenticated_false(self) -> None:
        creds = AuthCredentials(username="wrong", password=_VALID_PASSWORD)
        result = validate_credentials_node(_state_with(creds))

        assert _creds_from(result).is_authenticated is False

    def test_returns_explicit_delta(self) -> None:
        """Pinned: observability-consistency lift — returns explicit delta,
        preserving other fields via replace()."""
        creds = AuthCredentials(
            username=_VALID_USERNAME,
            password=_VALID_PASSWORD,
            message="prior",
        )
        result = validate_credentials_node(_state_with(creds))

        assert "credentials" in result
        assert isinstance(_creds_from(result), AuthCredentials)
        # message preserved through replace()
        assert _creds_from(result).message == "prior"

    def test_raises_when_credentials_missing_from_state(self) -> None:
        with pytest.raises(
            ValueError, match="validate_credentials_node requires credentials in state"
        ):
            validate_credentials_node({})


class TestSuccessNode:

    def test_stamps_success_message(self) -> None:
        """Pinned: success_node stamps the exact V2 success message."""
        creds = AuthCredentials(
            username="alice", password="pw", is_authenticated=True
        )
        result = success_node(_state_with(creds))

        assert _creds_from(result).message == _SUCCESS_MESSAGE

    def test_returns_explicit_delta(self) -> None:
        creds = AuthCredentials(username="alice", is_authenticated=True)
        result = success_node(_state_with(creds))

        assert "credentials" in result
        assert isinstance(_creds_from(result), AuthCredentials)

    def test_preserves_other_fields(self) -> None:
        creds = AuthCredentials(
            username="alice", password="pw", is_authenticated=True
        )
        result = success_node(_state_with(creds))

        assert _creds_from(result).username == "alice"
        assert _creds_from(result).password == "pw"
        assert _creds_from(result).is_authenticated is True

    def test_raises_when_credentials_missing_from_state(self) -> None:
        with pytest.raises(ValueError, match="success_node requires credentials in state"):
            success_node({})


class TestFailureNode:

    def test_stamps_failure_message(self) -> None:
        """Pinned: failure_node stamps the exact V2 failure message."""
        creds = AuthCredentials(username="alice", password="bad", is_authenticated=False)
        result = failure_node(_state_with(creds))

        assert _creds_from(result).message == _FAILURE_MESSAGE

    def test_clears_username_to_none(self) -> None:
        """Pinned: username cleared so retry loop prompts for it again."""
        creds = AuthCredentials(username="alice", password="bad", is_authenticated=False)
        result = failure_node(_state_with(creds))

        assert _creds_from(result).username is None

    def test_clears_password_to_none(self) -> None:
        """Pinned: password cleared independently of username."""
        creds = AuthCredentials(username="alice", password="bad", is_authenticated=False)
        result = failure_node(_state_with(creds))

        assert _creds_from(result).password is None

    def test_sets_is_authenticated_false(self) -> None:
        creds = AuthCredentials(username="alice", password="bad")
        result = failure_node(_state_with(creds))

        assert _creds_from(result).is_authenticated is False

    def test_returns_explicit_delta(self) -> None:
        creds = AuthCredentials(username="alice", password="bad")
        result = failure_node(_state_with(creds))

        assert "credentials" in result
        assert isinstance(_creds_from(result), AuthCredentials)

    def test_raises_when_credentials_missing_from_state(self) -> None:
        with pytest.raises(ValueError, match="failure_node requires credentials in state"):
            failure_node({})
