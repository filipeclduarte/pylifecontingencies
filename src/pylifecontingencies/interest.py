from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class InterestRate:
    """Annual effective interest rate with derived quantities and conversions."""

    i: float

    def __post_init__(self) -> None:
        if self.i <= -1:
            raise ValueError(f"Interest rate must be > -1, got {self.i}")

    # ------------------------------------------------------------------ #
    # Alternative constructors                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_delta(cls, delta: float) -> InterestRate:
        """Construct from force of interest δ."""
        return cls(i=float(np.expm1(delta)))

    @classmethod
    def from_discount(cls, d: float) -> InterestRate:
        """Construct from annual effective discount rate d = i/(1+i)."""
        if not (0 < d < 1):
            raise ValueError(f"Discount rate must be in (0,1), got {d}")
        return cls(i=d / (1.0 - d))

    @classmethod
    def from_nominal_interest(cls, i_m: float, m: int) -> InterestRate:
        """Construct from nominal interest rate i^(m) convertible m-thly."""
        return cls(i=float((1.0 + i_m / m) ** m - 1.0))

    @classmethod
    def from_nominal_discount(cls, d_m: float, m: int) -> InterestRate:
        """Construct from nominal discount rate d^(m) convertible m-thly."""
        return cls(i=float((1.0 - d_m / m) ** (-m) - 1.0))

    # ------------------------------------------------------------------ #
    # Derived quantities                                                   #
    # ------------------------------------------------------------------ #

    @property
    def v(self) -> float:
        """Annual discount factor v = 1/(1+i)."""
        return 1.0 / (1.0 + self.i)

    @property
    def d(self) -> float:
        """Annual effective discount rate d = i/(1+i) = 1 - v."""
        return self.i / (1.0 + self.i)

    @property
    def delta(self) -> float:
        """Force of interest δ = ln(1+i)."""
        return float(np.log1p(self.i))

    def i_m(self, m: int) -> float:
        """Nominal interest rate i^(m) convertible m-thly: i^(m) = m*((1+i)^(1/m) - 1)."""
        return float(m * ((1.0 + self.i) ** (1.0 / m) - 1.0))

    def d_m(self, m: int) -> float:
        """Nominal discount rate d^(m) convertible m-thly: d^(m) = m*(1 - (1+i)^(-1/m))."""
        return float(m * (1.0 - (1.0 + self.i) ** (-1.0 / m)))

    # ------------------------------------------------------------------ #
    # Accumulation / discounting                                           #
    # ------------------------------------------------------------------ #

    def accumulate(self, t: float) -> float:
        """Accumulation factor (1+i)^t."""
        return float((1.0 + self.i) ** t)

    def discount_factor(self, t: float) -> float:
        """Discount factor v^t = (1+i)^(-t)."""
        return float((1.0 + self.i) ** (-t))

    # ------------------------------------------------------------------ #
    # Conversions                                                          #
    # ------------------------------------------------------------------ #

    def real_rate(self, inflation: float) -> float:
        """Real interest rate via Fisher equation: (1+i_real) = (1+i)/(1+inflation)."""
        return (1.0 + self.i) / (1.0 + inflation) - 1.0

    def with_rate(self, i: float) -> InterestRate:
        """Return a new InterestRate with a different annual effective rate."""
        return InterestRate(i=i)

    # ------------------------------------------------------------------ #
    # UDD helpers used by actuarial functions                              #
    # ------------------------------------------------------------------ #

    def alpha(self, m: int) -> float:
        """α(m) = id / (i^(m) d^(m)) — UDD annuity conversion factor."""
        i_m = self.i_m(m)
        d_m = self.d_m(m)
        return (self.i * self.d) / (i_m * d_m)

    def beta(self, m: int) -> float:
        """β(m) = (i - i^(m)) / (i^(m) d^(m)) — UDD annuity conversion factor."""
        i_m = self.i_m(m)
        d_m = self.d_m(m)
        return (self.i - i_m) / (i_m * d_m)

    def __repr__(self) -> str:
        return f"InterestRate(i={self.i:.6g})"


# ------------------------------------------------------------------ #
# Module-level convenience functions matching R lifecontingencies     #
# ------------------------------------------------------------------ #

def convertible2Effective(i_m: float, m: int) -> float:
    """Convert nominal interest rate i^(m) to annual effective rate."""
    return (1.0 + i_m / m) ** m - 1.0


def effective2Convertible(i: float, m: int) -> float:
    """Convert annual effective rate to nominal rate i^(m)."""
    return m * ((1.0 + i) ** (1.0 / m) - 1.0)


def discount2Interest(d: float) -> float:
    """d → i."""
    return d / (1.0 - d)


def interest2Discount(i: float) -> float:
    """i → d."""
    return i / (1.0 + i)


def intensity2Interest(delta: float) -> float:
    """Force of interest δ → annual effective rate i."""
    return float(np.expm1(delta))


def interest2Intensity(i: float) -> float:
    """Annual effective rate i → force of interest δ."""
    return float(np.log1p(i))


def nominal2Real(i_nominal: float, inflation: float) -> float:
    """Fisher equation: nominal and inflation rate → real rate."""
    return (1.0 + i_nominal) / (1.0 + inflation) - 1.0


def real2Nominal(i_real: float, inflation: float) -> float:
    """Fisher equation: real rate and inflation → nominal rate."""
    return (1.0 + i_real) * (1.0 + inflation) - 1.0
