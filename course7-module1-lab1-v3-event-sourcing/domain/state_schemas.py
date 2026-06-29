from typing import TypedDict

from domain.auth_credentials import AuthCredentials
from domain.qa_exchange import QAExchange


class QAState(TypedDict, total=False):
    """LangGraph state schema for the QA workflow.

    Carries one domain object — the 'state as a single domain object
    field' pattern. The TypedDict declares exactly one key; nodes use
    `dataclasses.replace()` to update the QAExchange and return
    `{"exchange": new_exchange}`. LangGraph's last-write-wins merge keeps
    the contract intact.

    `AuthState` lives alongside this declaration; the V3 series is
    terminal at V3b and CounterState was scoped out.
    """

    exchange: QAExchange


class AuthState(TypedDict, total=False):
    """LangGraph state schema for the Auth workflow.

    Same single-key shape as QAState — one domain object per state.
    Nodes use replace() to update the AuthCredentials and return
    {"credentials": new_credentials}.
    """

    credentials: AuthCredentials