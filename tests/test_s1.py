import casadi as cs
import pytest

from liecasadi import S1, S1Tangent


def test_identity():
    assert S1.Identity().as_angle() == 0.0


def test_mul():
    a = S1(cs.pi / 4)
    b = S1(cs.pi / 4)
    c = a * b
    assert float(c.as_angle()) == pytest.approx(float(cs.pi / 2))


def test_sub_and_log_exp():
    a = S1(cs.pi / 3)
    b = S1(cs.pi / 6)
    diff = a - b
    rebuilt = diff.exp() * b
    assert float(rebuilt.as_angle()) == pytest.approx(float(cs.pi / 3))

    tangent = a.log()
    assert float(tangent.exp().as_angle()) == pytest.approx(float(cs.pi / 3))
