"""
Prospective and retrospective reserves via composition.
"""

from __future__ import annotations

from .actuarialtable import ActuarialTable
from .actuarial import axn, Axn, AExn, Exn
from .premiums import net_premium


def prospective_reserve(
    at: ActuarialTable,
    x: int,
    n: int,
    t: int,
    k: int = 1,
    benefit: str = "term",
    premium: float | None = None,
) -> float:
    """
    Prospective net reserve _t V at duration t for a policy issued at age x.

    _t V = APV(future benefits at x+t) - P * APV(future premiums at x+t)

    Parameters
    ----------
    at : ActuarialTable
    x : int
        Issue age.
    n : int
        Original policy term.
    t : int
        Duration at which to evaluate the reserve (0 ≤ t < n).
    k : int
        Payment frequency.
    benefit : {'term', 'whole', 'endowment'}
    premium : float or None
        Annual net premium; if None it is computed by net_premium().
    """
    if t >= n:
        return 0.0

    remaining = n - t
    x_t = x + t

    if premium is None:
        premium = net_premium(at, x=x, n=n, k=k, benefit=benefit)

    if benefit == "endowment":
        apv_future_benefits = AExn(at, x_t, n=remaining, k=k)
    else:
        apv_future_benefits = Axn(at, x_t, n=remaining, k=k)

    apv_future_premiums = axn(at, x_t, n=remaining, k=k)
    return apv_future_benefits - premium * apv_future_premiums


def retrospective_reserve(
    at: ActuarialTable,
    x: int,
    n: int,
    t: int,
    k: int = 1,
    benefit: str = "term",
    premium: float | None = None,
) -> float:
    """
    Retrospective net reserve _t V at duration t.

    _t V = (P * ä_{x:t|} - A^1_{x:t|}) / _t E_x

    Parameters
    ----------
    Same as prospective_reserve.
    """
    if t == 0:
        return 0.0

    if premium is None:
        premium = net_premium(at, x=x, n=n, k=k, benefit=benefit)

    apv_past_premiums = axn(at, x, n=t, k=k)
    apv_past_benefits = Axn(at, x, n=t, k=k)
    nEx = Exn(at, x, n=t)

    if nEx == 0.0:
        return 0.0
    return (premium * apv_past_premiums - apv_past_benefits) / nEx


def reserve_recursion(
    at: ActuarialTable,
    x: int,
    n: int,
    k: int = 1,
    benefit: str = "term",
) -> list[float]:
    """
    Full reserve vector [_0 V, _1 V, ..., _{n-1} V, _n V] by prospective formula.
    """
    P = net_premium(at, x=x, n=n, k=k, benefit=benefit)
    return [
        prospective_reserve(at, x=x, n=n, t=t, k=k, benefit=benefit, premium=P)
        for t in range(n + 1)
    ]
