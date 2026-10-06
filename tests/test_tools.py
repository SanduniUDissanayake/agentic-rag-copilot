import pytest

from tools import calculate


def test_subtraction():
    assert float(calculate("4.6 - 1.4")) == pytest.approx(3.2)


def test_percentage_decrease():
    assert float(calculate("(4.6 - 1.4) / 4.6 * 100")) == pytest.approx(69.565, rel=1e-3)


def test_negative_numbers():
    assert float(calculate("-5 + 2")) == pytest.approx(-3)


def test_rejects_code_injection():
    with pytest.raises(ValueError):
        calculate("__import__('os').system('echo hi')")


def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        calculate("1 / 0")