from typing import TypedDict

from domain.auth_credentials import AuthCredentials
from domain.qa_exchange import QAExchange


class QAState(TypedDict, total=False):
    """LangGraph state schema for the QA workflow.

    Carries one domain object — the V2 'state as a single domain object
    field' pattern. The TypedDict declares exactly one key; nodes use
    `dataclasses.replace()` to update the QAExchange and return
    `{"exchange": new_exchange}`. LangGraph's last-write-wins merge keeps
    the contract intact.

    `AuthState` and `CounterState` land here when V3b and V3c lift their
    respective workflows.
    """

    exchange: QAExchange


class AuthState(TypedDict, total=False):
    """LangGraph state schema for the Auth workflow.

    Same single-key shape as QAState — one domain object per state.
    Nodes use replace() to update the AuthCredentials and return
    {"credentials": new_credentials}.

    CounterState lands when V3c lifts the Counter workflow.
    """

    credentials: AuthCredentials