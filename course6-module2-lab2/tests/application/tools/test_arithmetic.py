from application.tools.arithmetic import add, subtract, multiply


def test_add():
    assert add.invoke({"a": 2, "b": 3}) == 5


def test_subtract():
    assert subtract.invoke({"a": 5, "b": 2}) == 3


def test_multiply():
    assert multiply.invoke({"a": 4, "b": 3}) == 12


def test_add_with_negatives():
    assert add.invoke({"a": -5, "b": 3}) == -2


def test_subtract_resulting_in_negative():
    assert subtract.invoke({"a": 2, "b": 5}) == -3