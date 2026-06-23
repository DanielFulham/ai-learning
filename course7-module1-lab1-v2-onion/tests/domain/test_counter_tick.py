from dataclasses import FrozenInstanceError

import pytest

from domain.counter_tick import CounterTick


def test_is_frozen_dataclass_cannot_be_mutated() -> None:
    tick = CounterTick(n=1, letter="a")
    with pytest.raises(FrozenInstanceError):
        setattr(tick, "n", 2)


def test_both_fields_accessible() -> None:
    tick = CounterTick(n=7, letter="m")
    assert tick.n == 7
    assert tick.letter == "m"


def test_minimum_n_of_one_satisfies_invariant() -> None:
    tick = CounterTick(n=1, letter="a")
    assert tick.n == 1
    assert tick.letter == "a"


def test_n_of_zero_raises() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        CounterTick(n=0, letter="a")


def test_letter_empty_string_raises() -> None:
    with pytest.raises(ValueError, match="single lowercase ASCII"):
        CounterTick(n=1, letter="")


def test_letter_uppercase_raises() -> None:
    with pytest.raises(ValueError, match="single lowercase ASCII"):
        CounterTick(n=1, letter="A")