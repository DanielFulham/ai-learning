"""Auth workflow event translator.

Translates one Auth node update (node_name + state_delta) into the
list of AuthEvents it represents. Pure function — same inputs produce
the same outputs except for event_id (uuid4 at translation time) and
occurred_at (clock at translation time).

Dispatch:
- InputNode → LoginAttempted (via _build_login_attempted codec helper)
- ValidateCredentialsNode (is_authenticated=True) → LoginSucceeded
- ValidateCredentialsNode (is_authenticated=False) → LoginFailed
- SuccessNode → [] (no event — UI-facing message stamp only)
- FailureNode → [] (no event — retry-state reset only)
- Unknown node → ValueError

Message handling: the translator hardcodes the event-log messages as
module-level constants (_LOGIN_SUCCEEDED_MESSAGE, _LOGIN_FAILED_MESSAGE)
rather than reading from credentials.message. The user-facing message
stamped by success_node/failure_node is a UI concern (what the user sees
on screen). The event log's message is an audit concern (what happened,
in standard wording). They happen to be similar in V3b but are not the
same field — coupling them through the state would be the wrong shape.
Same instinct as V3a separating user-safe answer from diagnostic
error_info. They may diverge without coupling failure.

Per-service duplication preserved per F04. No TranslationSpec, no
dispatch framework, no abstraction over QA and Auth translators.

Sensitive-field policy (F03) enforced as belt-and-braces: C1's
structural absence of password from LoginAttempted (braces), and the
codec helper as the only production construction site for Auth events
(belt). The helper never touches credentials.password.
"""
from datetime import datetime
from typing import Any, Callable
from uuid import UUID, uuid4

from application.graph_builders import (
    AUTH_FAILURE_NODE_NAME,
    AUTH_INPUT_NODE_NAME,
    AUTH_SUCCESS_NODE_NAME,
    AUTH_VALIDATE_CREDENTIALS_NODE_NAME,
)
from domain.auth_credentials import AuthCredentials
from domain.events.auth_events import (
    AuthEvent,
    LoginAttempted,
    LoginFailed,
    LoginSucceeded,
)


_LOGIN_SUCCEEDED_MESSAGE = "Authentication succeeded."
_LOGIN_FAILED_MESSAGE = "Authentication failed."


def _build_login_attempted(
    credentials: AuthCredentials,
    run_id: UUID,
    clock: Callable[[], datetime],
) -> LoginAttempted:
    """Construct LoginAttempted with sensitive-field policy applied.

    F03's procedural belt over C1's structural braces: password is
    structurally absent from LoginAttempted's dataclass (C1), AND this
    helper is the only construction site for LoginAttempted in
    production code (C3). The helper deliberately does NOT touch
    credentials.password — it never appears in this function body.

    Username narrowing: AuthCredentials.username is str | None on the
    carrier; LoginAttempted.username is str (non-nullable, C1
    strongest-contract decision). Translator dispatch guarantees this
    helper only runs when InputNode has populated username — raise
    ValueError if violated, same shape as the QA translator's
    malformed-delta guard.
    """
    if credentials.username is None:
        raise ValueError(
            "LoginAttempted requires credentials.username; "
            "InputNode must populate it before ValidateCredentialsNode"
        )
    return LoginAttempted(
        event_id=uuid4(),
        aggregate_id=run_id,
        occurred_at=clock(),
        username=credentials.username,
    )


def _build_login_succeeded(
    credentials: AuthCredentials,
    run_id: UUID,
    clock: Callable[[], datetime],
) -> LoginSucceeded:
    """Construct LoginSucceeded with the audit-log message.

    The message is the hardcoded audit-log constant, not credentials.message.
    See module docstring on message handling.
    """
    if credentials.username is None:
        raise ValueError(
            "LoginSucceeded requires credentials.username; "
            "ValidateCredentialsNode must receive a populated username"
        )
    return LoginSucceeded(
        event_id=uuid4(),
        aggregate_id=run_id,
        occurred_at=clock(),
        username=credentials.username,
        message=_LOGIN_SUCCEEDED_MESSAGE,
    )


def _build_login_failed(
    credentials: AuthCredentials,
    run_id: UUID,
    clock: Callable[[], datetime],
) -> LoginFailed:
    """Construct LoginFailed with the audit-log message.

    The message is the hardcoded audit-log constant, not credentials.message.
    See module docstring on message handling.
    """
    if credentials.username is None:
        raise ValueError(
            "LoginFailed requires credentials.username; "
            "ValidateCredentialsNode must receive a populated username"
        )
    return LoginFailed(
        event_id=uuid4(),
        aggregate_id=run_id,
        occurred_at=clock(),
        username=credentials.username,
        message=_LOGIN_FAILED_MESSAGE,
    )


def translate_auth_update(
    node_name: str,
    state_delta: dict[str, Any],
    run_id: UUID,
    clock: Callable[[], datetime],
) -> list[AuthEvent]:
    """Translate one Auth node update into the events it represents.

    Pure function: same inputs produce the same outputs except for event_id
    (uuid4 at translation time) and occurred_at (clock at translation time).
    Both timing concerns are explicit — translator is the canonical source
    per the V3a pin.

    The return type is list[AuthEvent] even when a node emits no events
    (SuccessNode, FailureNode return []). The list shape keeps the contract
    uniform across all translation boundaries.

    Two-stage narrowing: the caller has already isinstance-checked the chunk
    against dict to access state_delta; this function performs the second
    narrow against AuthCredentials. Missing or wrong-type credentials raise.

    Unknown node names raise. The Auth workflow is enumerated and small;
    a surprising node name signals the graph changed without the translator
    being updated.
    """
    credentials = state_delta.get("credentials")
    if not isinstance(credentials, AuthCredentials):
        raise ValueError(
            f"Auth state delta from node '{node_name}' missing or malformed "
            f"'credentials' field; expected AuthCredentials, got "
            f"{type(credentials).__name__}"
        )

    if node_name == AUTH_INPUT_NODE_NAME:
        return [_build_login_attempted(credentials, run_id, clock)]

    if node_name == AUTH_VALIDATE_CREDENTIALS_NODE_NAME:
        if credentials.is_authenticated is True:
            return [_build_login_succeeded(credentials, run_id, clock)]
        if credentials.is_authenticated is False:
            return [_build_login_failed(credentials, run_id, clock)]
        raise ValueError(
            f"ValidateCredentialsNode update has is_authenticated=None — "
            f"the node must set is_authenticated to True or False"
        )

    if node_name == AUTH_SUCCESS_NODE_NAME:
        return []

    if node_name == AUTH_FAILURE_NODE_NAME:
        return []

    raise ValueError(
        f"Unknown auth node '{node_name}' — translator handles "
        f"'{AUTH_INPUT_NODE_NAME}', '{AUTH_VALIDATE_CREDENTIALS_NODE_NAME}', "
        f"'{AUTH_SUCCESS_NODE_NAME}', and '{AUTH_FAILURE_NODE_NAME}' only"
    )
