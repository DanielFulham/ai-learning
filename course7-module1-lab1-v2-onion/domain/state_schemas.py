from typing import TypedDict

from domain.auth_credentials import AuthCredentials
from domain.counter_tick import CounterTick
from domain.qa_exchange import QAExchange


class AuthState(TypedDict, total=False):
    """LangGraph state for the Auth workflow.

    Holds a single domain field — the credentials object carrying the
    verdict and message. Closes the loose-bag pattern from V1 where
    is_authenticated, username, password, and output were independent
    optional keys with no invariant linking them.
    """

    credentials: AuthCredentials


class QAState(TypedDict, total=False):
    """LangGraph state for the QA workflow.

    Holds a single domain field — the question-context-answer exchange.
    Closes V1's `{"valid": False, "error": ...}` overflow where nodes
    wrote keys the TypedDict didn't declare.
    """

    exchange: QAExchange


class CounterState(TypedDict, total=False):
    """LangGraph state for the Counter workflow.

    Holds a single domain field — the current tick. Absence of the key
    is the "no tick yet" state; presence is "a tick has happened with
    these values".
    """

    tick: CounterTick