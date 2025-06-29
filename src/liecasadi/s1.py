import dataclasses
import casadi as cs

from liecasadi.hints import Angle, Scalar


def _wrap_to_pi(angle: Scalar) -> Scalar:
    """Wrap an angle to the range [-pi, pi]."""
    return cs.fmod(angle + cs.pi, 2 * cs.pi) - cs.pi


@dataclasses.dataclass
class S1:
    """Class representing the circle manifold S1."""

    angle: Angle

    def __repr__(self) -> str:
        return f"S1 angle: {self.angle}"

    @staticmethod
    def Identity() -> "S1":
        """Return the identity element (0 radians)."""
        return S1(0.0)

    def as_angle(self) -> Angle:
        """Return the angle value."""
        return self.angle

    def inverse(self) -> "S1":
        """Return the inverse rotation."""
        return S1(-self.angle)

    def __mul__(self, other: "S1") -> "S1":
        if type(self) is type(other):
            return S1(_wrap_to_pi(self.angle + other.angle))
        raise RuntimeError("[S1: __mul__] Please multiply with an S1 object.")

    def __rmul__(self, other: "S1") -> "S1":
        return self.__mul__(other)

    def log(self) -> "S1Tangent":
        """Return the tangent vector via the logarithm map."""
        return S1Tangent(_wrap_to_pi(self.angle))

    def __sub__(self, other: "S1") -> "S1Tangent":
        if type(self) is type(other):
            return S1Tangent(_wrap_to_pi(self.angle - other.angle))
        raise RuntimeError("[S1: __sub__] Please subtract an S1 object.")


@dataclasses.dataclass
class S1Tangent:
    """Tangent space element of S1."""

    val: Scalar

    def __repr__(self) -> str:
        return f"S1Tangent: {self.val}"

    def exp(self) -> S1:
        """Exponential map returning an S1 element."""
        return S1(_wrap_to_pi(self.val))

    def __add__(self, other: S1) -> S1:
        if type(other) is S1:
            return S1(_wrap_to_pi(other.angle + self.val))
        raise RuntimeError("[S1Tangent: __add__] Please add an S1 object.")

    def __radd__(self, other: S1) -> S1:
        return self.__add__(other)

    def __mul__(self, scalar: float) -> "S1Tangent":
        if isinstance(scalar, (int, float)):
            return S1Tangent(self.val * scalar)
        raise RuntimeError("[S1Tangent: __mul__] Please multiply with a scalar.")

    def __rmul__(self, scalar: float) -> "S1Tangent":
        return self.__mul__(scalar)

    def value(self) -> Scalar:
        """Return the underlying scalar value."""
        return self.val
