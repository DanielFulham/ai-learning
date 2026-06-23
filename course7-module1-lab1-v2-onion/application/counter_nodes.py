import random
import string

from domain.counter_tick import CounterTick
from domain.state_schemas import CounterState


def add_node(state: CounterState) -> CounterState:
    """Advance the counter by one tick.

    Reads the current tick (or treats absence as n=0), increments n,
    selects a random lowercase letter, and returns a fresh CounterTick.
    The domain invariant `n >= 1` is satisfied by construction — this
    node only ever produces ticks with n at least 1.
    """
    current = state.get("tick")
    next_n = (current.n if current is not None else 0) + 1
    next_letter = random.choice(string.ascii_lowercase)
    return {"tick": CounterTick(n=next_n, letter=next_letter)}