"""
Interest-theory financial functions (no life table dependency).
Matches the R lifecontingencies exports: annuity, presentValue,
accumulatedValue, increasingAnnuity, decreasingAnnuity, duration, convexity.
"""

from __future__ import annotations

import numpy as np


def presentValue(
    cashflows: list[float] | np.ndarray,
    timepoints: list[float] | np.ndarray,
    i: float,
) -> float:
    """
    Present value of a series of deterministic cash flows.

    Parameters
    ----------
    cashflows : array-like
        Cash flow amounts.
    timepoints : array-like
        Payment times (in years from t=0).
    i : float
        Annual effective interest rate.
    """
    cf = np.asarray(cashflows, dtype=float)
    t = np.asarray(timepoints, dtype=float)
    v = 1.0 / (1.0 + i)
    return float(np.sum(cf * v ** t))


def accumulatedValue(
    cashflows: list[float] | np.ndarray,
    timepoints: list[float] | np.ndarray,
    i: float,
) -> float:
    """Accumulated value of cash flows at the time of the last payment."""
    cf = np.asarray(cashflows, dtype=float)
    t = np.asarray(timepoints, dtype=float)
    T = t[-1]
    return float(np.sum(cf * (1.0 + i) ** (T - t)))


def annuity(
    i: float,
    n: int | float,
    *,
    k: int = 1,
    type: str = "due",
) -> float:
    """
    Present value of a unit annuity (no mortality).

    Parameters
    ----------
    i : float
        Annual effective interest rate.
    n : int or float
        Term in years. Use n=float('inf') for perpetuity.
    k : int
        Number of payments per year (k=1 annual, k=2 semi-annual, ...).
    type : {'due', 'immediate'}
        Annuity-due (payments at start of period) or annuity-immediate
        (payments at end of period).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    i_m = k * ((1.0 + i) ** (1.0 / k) - 1.0)  # i^(m)
    v_k = 1.0 / (1.0 + i_m / k)               # v^(1/k)

    if np.isinf(n):
        # perpetuity
        if type == "due":
            return (1.0 + i_m / k) / i_m
        return 1.0 / i_m

    payments = n * k
    # annuity-immediate (payments at end of each 1/k period)
    pv_imm = (1.0 - v_k ** payments) / i_m if i_m != 0 else float(payments) / k
    if type == "immediate":
        return float(pv_imm)
    # annuity-due = annuity-immediate * (1 + i^(m)/m)
    return float(pv_imm * (1.0 + i_m / k))


def increasingAnnuity(i: float, n: int, *, type: str = "due") -> float:
    """PV of an annually-increasing annuity (1, 2, ..., n) — no mortality."""
    if type == "due":
        return float(
            ((1.0 + i) * (annuity(i, n, type="due") - n * (1.0 / (1.0 + i)) ** n)) / i
        )
    # immediate
    return float(
        (annuity(i, n, type="immediate") - n * (1.0 / (1.0 + i)) ** n) / i
    )


def decreasingAnnuity(i: float, n: int, *, type: str = "due") -> float:
    """PV of a decreasing annuity (n, n-1, ..., 1) — no mortality."""
    return float((n + (1 if type == "due" else 0)) * annuity(i, n, type=type)
                 - increasingAnnuity(i, n, type=type))


def duration(
    cashflows: list[float] | np.ndarray,
    timepoints: list[float] | np.ndarray,
    i: float,
    *,
    modified: bool = False,
) -> float:
    """
    Macaulay (or modified) duration of a cash-flow stream.

    Parameters
    ----------
    modified : bool
        If True return modified duration = Macaulay / (1+i).
    """
    cf = np.asarray(cashflows, dtype=float)
    t = np.asarray(timepoints, dtype=float)
    v = 1.0 / (1.0 + i)
    pv = cf * v ** t
    mac = float(np.sum(t * pv) / np.sum(pv))
    return mac / (1.0 + i) if modified else mac


def convexity(
    cashflows: list[float] | np.ndarray,
    timepoints: list[float] | np.ndarray,
    i: float,
) -> float:
    """Convexity of a cash-flow stream."""
    cf = np.asarray(cashflows, dtype=float)
    t = np.asarray(timepoints, dtype=float)
    v = 1.0 / (1.0 + i)
    pv = cf * v ** t
    return float(np.sum(t * (t + 1) * pv) / (np.sum(pv) * (1.0 + i) ** 2))
