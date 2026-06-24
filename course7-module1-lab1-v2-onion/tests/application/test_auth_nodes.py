from unittest.mock import MagicMock

import pytest

from application.auth_nodes import (
    failure_node,
    make_input_node,
    success_node,
    validate_credentials_node,
)
from domain.auth_credentials import AuthCredentials
from domain.state_schemas import AuthState
from interfaces.input_provider_interface import InputProviderInterface


def _make_input_provider(*responses: str) -> MagicMock:
    provider = MagicMock(spec=InputProviderInterface)
    provider.prompt.side_effect = list(responses)
    return provider


def test_input_node_prompts_for_both_when_no_username() -> None:
    provider = _make_input_provider("test_user", "secure_password")
    node = make_input_node(provider)
    result = node({})
    credentials = result.get("credentials")
    assert credentials is not None
    assert provider.prompt.call_count == 2
    assert credentials.username == "test_user"
    assert credentials.password == "secure_password"


def test_input_node_skips_username_prompt_when_present() -> None:
    provider = _make_input_provider("secure_password")
    node = make_input_node(provider)
    state: AuthState = {"credentials": AuthCredentials(username="test_user")}
    result = node(state)
    credentials = result.get("credentials")
    assert credentials is not None
    assert provider.prompt.call_count == 1
    assert credentials.username == "test_user"
    assert credentials.password == "secure_password"


def test_validate_returns_true_for_canonical_credentials() -> None:
    state: AuthState = {
        "credentials": AuthCredentials(username="test_user", password="secure_password")
    }
    result = validate_credentials_node(state)
    credentials = result.get("credentials")
    assert credentials is not None
    assert credentials.is_authenticated is True


def test_validate_returns_false_for_wrong_password() -> None:
    state: AuthState = {
        "credentials": AuthCredentials(username="test_user", password="wrong")
    }
    result = validate_credentials_node(state)
    credentials = result.get("credentials")
    assert credentials is not None
    assert credentials.is_authenticated is False


def test_validate_returns_false_for_unknown_username() -> None:
    state: AuthState = {
        "credentials": AuthCredentials(username="someone_else", password="secure_password")
    }
    result = validate_credentials_node(state)
    credentials = result.get("credentials")
    assert credentials is not None
    assert credentials.is_authenticated is False


def test_success_node_stamps_success_message() -> None:
    state: AuthState = {
        "credentials": AuthCredentials(
            username="test_user", password="secure_password", is_authenticated=True
        )
    }
    result = success_node(state)
    credentials = result.get("credentials")
    assert credentials is not None
    assert credentials.message == "Authentication successful! Welcome."
    assert credentials.is_authenticated is True


def test_failure_node_clears_both_username_and_password_and_stamps_message() -> None:
    """Clearing username matters — without it, a user who accidentally entered the
    username on the first try has no opportunity to correct it. InputNode
    skips the username prompt whenever a username is present in state."""
    state: AuthState = {
        "credentials": AuthCredentials(
            username="test_user", password="wrong", is_authenticated=False
        )
    }
    result = failure_node(state)
    credentials = result.get("credentials")
    assert credentials is not None
    assert credentials.username is None
    assert credentials.password is None
    assert credentials.is_authenticated is False
    assert credentials.message == "Not Successful, please try again!"


def test_input_node_raises_when_provider_exhausted() -> None:
    """The IndexError from ScriptedInputProvider exhaustion surfaces through
    the node rather than being swallowed."""
    provider = MagicMock(spec=InputProviderInterface)
    provider.prompt.side_effect = IndexError("exhausted")
    node = make_input_node(provider)
    with pytest.raises(IndexError, match="exhausted"):
        node({})
