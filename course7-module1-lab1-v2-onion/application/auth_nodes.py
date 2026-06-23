from dataclasses import replace

from domain.auth_credentials import AuthCredentials
from domain.state_schemas import AuthState
from interfaces.input_provider_interface import InputProviderInterface


def make_input_node(input_provider: InputProviderInterface):
    """Factory closing over the input provider.

    Returns a node function with the LangGraph signature `(state) -> dict`.
    The closure captures the provider so the node can prompt for username
    and password without importing infra or calling `input()` directly.
    Closes V1 finding #5.
    """

    def input_node(state: AuthState) -> AuthState:
        current = state.get("credentials") or AuthCredentials()

        if current.username is None or current.username == "":
            username = input_provider.prompt("What is your username? ")
        else:
            username = current.username

        password = input_provider.prompt("Enter your password: ")

        return {"credentials": AuthCredentials(username=username, password=password)}

    return input_node


def validate_credentials_node(state: AuthState) -> AuthState:
    """Verify credentials against the canonical test pair.

    Pure function — no provider dependency. Returns a new
    `AuthCredentials` with `is_authenticated` set; the existing username
    and password carry through via `dataclasses.replace`.
    """
    current = state.get("credentials") or AuthCredentials()

    is_authenticated = (
        current.username == "test_user" and current.password == "secure_password"
    )

    return {
        "credentials": replace(current, is_authenticated=is_authenticated),
    }


def success_node(state: AuthState) -> AuthState:
    """Stamp the verdict message on a successful credentials object."""
    current = state.get("credentials") or AuthCredentials()
    return {
        "credentials": replace(current, message="Authentication successful! Welcome."),
    }


def failure_node(state: AuthState) -> AuthState:
    """Stamp the verdict message on a failed credentials object, clearing
    both username and password so the next loop iteration prompts cleanly
    for both. Without clearing username, a user who put their
    username on the first try has no opportunity to correct it — InputNode
    skips the username prompt whenever a username is present in state.
    """
    return {
        "credentials": AuthCredentials(
            username=None,
            password=None,
            is_authenticated=False,
            message="Not Successful, please try again!",
        ),
    }