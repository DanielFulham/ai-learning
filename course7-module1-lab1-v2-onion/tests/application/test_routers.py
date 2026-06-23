from application.routers import auth_router, counter_stop_router
from domain.auth_credentials import AuthCredentials
from domain.counter_tick import CounterTick
from domain.state_schemas import AuthState, CounterState


def test_auth_router_returns_success_for_authenticated_credentials() -> None:
    state: AuthState = {
        "credentials": AuthCredentials(
            username="test_user", is_authenticated=True
        )
    }
    assert auth_router(state) == "success"


def test_auth_router_returns_failure_for_unauthenticated_credentials() -> None:
    state: AuthState = {
        "credentials": AuthCredentials(
            username="test_user", password="wrong", is_authenticated=False
        )
    }
    assert auth_router(state) == "failure"


def test_auth_router_returns_failure_when_credentials_absent() -> None:
    """Defensive — `validate_credentials_node` always writes credentials,
    so this path shouldn't fire in production. Pinned here so changing the
    behaviour to raise becomes a test change, not a silent semantics shift."""
    assert auth_router({}) == "failure"


def test_auth_router_returns_failure_when_is_authenticated_is_none() -> None:
    """A credentials object with no verdict yet (None) is not a success."""
    state: AuthState = {
        "credentials": AuthCredentials(username="test_user", password="anything")
    }
    assert auth_router(state) == "failure"


def test_counter_router_returns_continue_below_threshold() -> None:
    state: CounterState = {"tick": CounterTick(n=12, letter="a")}
    assert counter_stop_router(state) == "continue"


def test_counter_router_returns_stop_at_threshold() -> None:
    """V1 behaviour preserved — termination at n >= 13."""
    state: CounterState = {"tick": CounterTick(n=13, letter="a")}
    assert counter_stop_router(state) == "stop"


def test_counter_router_returns_stop_above_threshold() -> None:
    state: CounterState = {"tick": CounterTick(n=100, letter="z")}
    assert counter_stop_router(state) == "stop"


def test_counter_router_returns_continue_when_tick_absent() -> None:
    """Initial state — no tick has happened yet, the loop should continue."""
    assert counter_stop_router({}) == "continue"