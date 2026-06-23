from application.counter_nodes import add_node
from domain.counter_tick import CounterTick
from domain.state_schemas import CounterState


def test_add_node_from_empty_state_produces_n_equals_one() -> None:
    result = add_node({})
    tick = result.get("tick")
    assert tick is not None
    assert tick.n == 1


def test_add_node_increments_existing_tick() -> None:
    state: CounterState = {"tick": CounterTick(n=5, letter="a")}
    result = add_node(state)
    tick = result.get("tick")
    assert tick is not None
    assert tick.n == 6


def test_add_node_produces_lowercase_letter() -> None:
    """The domain invariant pins this — verifying the node respects it."""
    for _ in range(20):
        result = add_node({})
        tick = result.get("tick")
        assert tick is not None
        assert tick.letter.islower()
        assert len(tick.letter) == 1


def test_add_node_returns_new_tick_instance() -> None:
    """No mutation of input state — the node returns a fresh CounterTick."""
    original = CounterTick(n=3, letter="z")
    state: CounterState = {"tick": original}
    result = add_node(state)
    tick = result.get("tick")
    assert tick is not None
    assert tick is not original
    assert tick.n == 4