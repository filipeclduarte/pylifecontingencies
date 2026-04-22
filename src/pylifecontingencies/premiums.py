"""
Net and gross premium calculations via the equivalence principle.
"""

from __future__ import annotations

from .actuarialtable import ActuarialTable
from .actuarial import axn, Axn, AExn


def net_premium(
    at: ActuarialTable,
    x: int,
    n: int | None = None,
    k: int = 1,
    benefit: str = "term",
) -> float:
    """
    Net (benefit) premium P via the equivalence principle.

    P = APV(benefits) / APV(annuity of premium payments)

    Parameters
    ----------
    at : ActuarialTable
    x : int
        Issue age.
    n : int or None
        Term in years. None = whole-life.
    k : int
        Payment frequency.
    benefit : {'term', 'whole', 'endowment'}
        Type of insurance benefit.
    """
    if benefit == "endowment":
        if n is None:
            raise ValueError("endowment requires a finite term n")
        apv_benefit = AExn(at, x, n=n, k=k)
    elif benefit in ("term", "whole"):
        apv_benefit = Axn(at, x, n=n, k=k)
    else:
        raise ValueError(f"Unknown benefit type: {benefit!r}")

    apv_annuity = axn(at, x, n=n, k=k)
    if apv_annuity == 0.0:
        return float("inf")
    return apv_benefit / apv_annuity


def gross_premium(
    at: ActuarialTable,
    x: int,
    n: int | None = None,
    k: int = 1,
    benefit: str = "term",
    expense_ratio: float = 0.0,
    per_policy: float = 0.0,
) -> float:
    """
    Gross premium including per-unit expense loading.

    G = (APV(benefits) + APV(per-policy expenses)) /
        ((1 - expense_ratio) * APV(premium annuity))

    Parameters
    ----------
    expense_ratio : float
        Proportion of each premium consumed by expenses (0 ≤ e < 1).
    per_policy : float
        Fixed expense per year of policy in force (paid when premium is paid).
    """
    if benefit == "endowment":
        if n is None:
            raise ValueError("endowment requires a finite term n")
        apv_benefit = AExn(at, x, n=n, k=k)
    else:
        apv_benefit = Axn(at, x, n=n, k=k)

    apv_annuity = axn(at, x, n=n, k=k)
    numerator = apv_benefit + per_policy * apv_annuity
    denominator = (1.0 - expense_ratio) * apv_annuity
    if denominator == 0.0:
        return float("inf")
    return numerator / denominator
