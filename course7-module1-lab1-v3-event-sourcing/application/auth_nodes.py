from dataclasses import replace

from domain.auth_credentials import AuthCredentials
from domain.state_schemas import AuthState
from interfaces.input_provider_interface import InputProviderInterface


_VALID_USERNAME = "test_user"
_VALID_PASSWORD = "secure_password"

_SUCCESS_MESSAGE = "Authentication successful! Welcome."
_FAILURE_MESSAGE = "Not Successful, please try again!"


def make_input_node(input_provider: InputProviderInterface):
    """Factory closing over the input provider.

    Returns a node function with the LangGraph signature `(state) -> dict`.
    The closure captures the provider so the node can prompt for username
    and password without importing infra or calling input() directly.

    V3b observability-consistency lift: always returns an explicit
    {"credentials": replace(current, username=..., password=...)} delta.
    The auth service pre-populates the state with AuthCredentials() before
    streaming, so credentials is always present.

    Username reuse: if credentials.username is already set (non-None and
    non-empty), the username prompt is skipped and the existing value is
    reused. This matches V2's behaviour — a retry loop after a failed
    login preserves whatever the failure_node wrote to the carrier
    (which clears username to None, so the prompt always fires on retry).
    """

    def input_node(state: AuthState) -> AuthState:
        current = state.get("credentials")
        if current is None:
            raise ValueError("input_node requires credentials in state")

        if current.username is None or current.username == "":
            username = input_provider.prompt("What is your username? ")
        else:
            username = current.username

        password = input_provider.prompt_secret("Enter your password: ")

        return {"credentials": replace(current, username=username, password=password)}

    return input_node


def validate_credentials_node(state: AuthState) -> AuthState:
    """Verify credentials against the canonical test pair.

    Pure function — no provider dependency. Returns an explicit
    {"credentials": replace(current, is_authenticated=...)} delta.

    V3b observability-consistency lift: always returns an explicit delta.
    The verdict is a bool — True for the matching test pair, False for
    everything else. The success_node and failure_node stamp the message;
    this node owns only the verdict.
    """
    current = state.get("credentials")
    if current is None:
        raise ValueError("validate_credentials_node requires credentials in state")

    is_authenticated = (
        current.username == _VALID_USERNAME and current.password == _VALID_PASSWORD
    )

    return {"credentials": replace(current, is_authenticated=is_authenticated)}


def success_node(state: AuthState) -> AuthState:
    """Stamp the success message on a successful credentials object.

    V3b observability-consistency lift: returns an explicit
    {"credentials": replace(current, message=...)} delta.
    No event emitted — the translator skips SuccessNode.
    The event-log message is owned by the translator's hardcoded constant;
    this node owns the UI-facing message.
    """
    current = state.get("credentials")
    if current is None:
        raise ValueError("success_node requires credentials in state")

    return {"credentials": replace(current, message=_SUCCESS_MESSAGE)}


def failure_node(state: AuthState) -> AuthState:
    """Stamp the failure message and clear username/password for retry.

    Clears both username and password so the next loop iteration prompts
    cleanly for both. Without clearing username, a user who entered their
    username on the first try has no opportunity to correct it —
    input_node skips the username prompt whenever a non-empty username is
    present in the carrier.

    V3b observability-consistency lift: returns an explicit
    {"credentials": replace(current, ...)} delta.
    No event emitted — the translator skips FailureNode.
    """
    current = state.get("credentials")
    if current is None:
        raise ValueError("failure_node requires credentials in state")

    return {
        "credentials": replace(
            current,
            username=None,
            password=None,
            is_authenticated=False,
            message=_FAILURE_MESSAGE,
        )
    }
