from domain.state_schemas import AuthState, CounterState


def auth_router(state: AuthState) -> str:
    """Route after credential validation.

    Returns string literals — `"success"` or `"failure"` — paired with a
    `dict[str, str]` path_map in the graph builder. Closes V1 finding #3
    on the Counter side; same shape adopted here for consistency. The
    `dict[bool, str]` pattern from V1's Counter is dropped entirely.
    """
    credentials = state.get("credentials")
    if credentials is None:
        return "failure"
    return "success" if credentials.is_authenticated else "failure"


def counter_stop_router(state: CounterState) -> str:
    """Route after each counter tick.

    Returns `"stop"` once the tick has reached the termination threshold
    (n >= 13, preserving V1's behaviour), `"continue"` otherwise. Threshold
    lives here because it's a routing decision, not a domain invariant —
    `CounterTick` itself allows any n >= 1.
    """
    tick = state.get("tick")
    if tick is None:
        return "continue"
    return "stop" if tick.n >= 13 else "continue"