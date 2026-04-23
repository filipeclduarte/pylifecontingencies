"""
Monte Carlo present value simulation for static actuarial tables.
"""

from __future__ import annotations

import numpy as np

from .actuarialtable import ActuarialTable
from .dynamic.stochastic import StochasticResult

_BENEFIT_ALIASES = {
    "term": "term",
    "term_insurance": "term",
    "whole": "insurance",
    "whole_life": "insurance",
    "whole_life_insurance": "insurance",
    "insurance": "insurance",
    "annuity": "annuity",
    "life_annuity": "annuity",
    "pure_endowment": "pure_endowment",
    "endowment": "endowment",
    "endowment_insurance": "endowment",
    "increasing": "increasing",
    "decreasing": "decreasing",
}


def _normalize_benefit(benefit: str) -> str:
    key = benefit.strip().lower()
    if key not in _BENEFIT_ALIASES:
        choices = ", ".join(sorted(_BENEFIT_ALIASES))
        raise ValueError(f"Unknown benefit type: {benefit!r}. Expected one of: {choices}")
    return _BENEFIT_ALIASES[key]


def _death_year_pmf(at: ActuarialTable, x: int) -> np.ndarray:
    """Return P[K_x = k] for k = 0, ..., omega-x-1."""
    ages = np.arange(x, at.omega)
    lx_x = at.lifetable.lx(x)
    dx = at.lifetable.dx(ages)
    return np.asarray(dx, dtype=float) / float(lx_x)


def _pv_annuity_kthly(
    K: np.ndarray,
    J: np.ndarray,
    v: float,
    k: int,
    n: int | None,
) -> np.ndarray:
    """
    PV of a k-thly life annuity-due under UDD.

    Under UDD, the complete future lifetime within the year of death is
    uniformly distributed, giving J = floor(k*S) ~ DiscreteUniform(0, k-1).
    The number of k-thly payments is  min(kK + J + 1, kn).

    For k=1, J=0 and this reduces to the standard annual formula.
    """
    v_k = v ** (1.0 / k)
    d_k = k * (1.0 - v_k)  # d^(k)

    if n is None:
        periods = k * K + J + 1
    else:
        periods = np.minimum(k * K + J + 1, k * int(n))

    if np.isclose(d_k, 0.0):
        return periods.astype(float) / k
    return (1.0 - v_k ** periods) / d_k


def _pv_from_death_years(
    at: ActuarialTable,
    death_years: np.ndarray,
    benefit: str,
    n: int | None,
) -> np.ndarray:
    """Map curtate future lifetimes to present values."""
    v = at.interest.v
    K = np.asarray(death_years, dtype=int)
    whole_life = n is None
    term = None if whole_life else int(n)

    if benefit == "annuity":
        J = np.zeros(len(K), dtype=int)
        return _pv_annuity_kthly(K, J, v, k=1, n=term)

    death_in_term = np.ones_like(K, dtype=bool) if whole_life else (K < term)

    if benefit in {"term", "insurance"}:
        return np.where(death_in_term, v ** (K + 1), 0.0)
    if benefit == "pure_endowment":
        if whole_life:
            raise ValueError("pure_endowment requires a finite term n")
        return np.where(K >= term, v ** term, 0.0)
    if benefit == "endowment":
        if whole_life:
            raise ValueError("endowment requires a finite term n")
        return np.where(K < term, v ** (K + 1), v ** term)
    if benefit == "increasing":
        return np.where(death_in_term, (K + 1) * v ** (K + 1), 0.0)
    if benefit == "decreasing":
        if whole_life:
            raise ValueError("decreasing insurance requires a finite term n")
        return np.where(K < term, (term - K) * v ** (K + 1), 0.0)

    raise ValueError(f"Unknown benefit type: {benefit!r}")


def simulate_pv(
    at: ActuarialTable,
    x: int,
    n: int | None = None,
    benefit: str = "term",
    n_sim: int = 10_000,
    random_state: int | np.random.Generator | None = None,
    k: int = 1,
    m: int = 0,
) -> StochasticResult:
    """
    Simulate a present-value distribution via Monte Carlo.

    Parameters
    ----------
    at : ActuarialTable
        Static actuarial table.
    x : int
        Attained age at issue.
    n : int or None
        Term in years. ``None`` gives a whole-life contract when supported.
    benefit : str
        One of ``"term"``, ``"whole"``, ``"insurance"``, ``"annuity"``,
        ``"pure_endowment"``, ``"endowment"``, ``"increasing"``,
        or ``"decreasing"``. Common aliases such as ``"whole_life"``
        and ``"term_insurance"`` are also accepted.
    n_sim : int
        Number of Monte Carlo draws.
    random_state : int, Generator, or None
        Seed or NumPy generator for reproducibility.
    k : int
        Payment frequency per year (default 1 = annual).  Values > 1 are
        supported for ``benefit='annuity'`` only; the UDD approximation is
        used to sample the fractional year of death.
    m : int
        Deferral period in years (default 0 = no deferral).  A positive ``m``
        causes the benefit to start only if the insured survives to age ``x+m``;
        scenarios where death occurs before ``m`` receive PV = 0.

    Examples
    --------
    >>> lt = load_table("soa_ilt")
    >>> at = ActuarialTable(lt, 0.03)
    >>> # monthly annuity
    >>> r = simulate_pv(at, x=40, n=20, benefit="annuity", k=12, n_sim=50_000, random_state=0)
    >>> # 10-year deferred whole-life annuity
    >>> r2 = simulate_pv(at, x=40, benefit="annuity", m=10, n_sim=50_000, random_state=0)
    """
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")
    if x < at.x_min or x >= at.omega:
        raise ValueError(f"Age {x} out of table range [{at.x_min}, {at.omega - 1}]")
    if n is not None and n <= 0:
        raise ValueError("n must be positive when supplied")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(m, (int, np.integer)) or m < 0:
        raise ValueError("m must be a non-negative integer")

    benefit = _normalize_benefit(benefit)

    if k > 1 and benefit != "annuity":
        raise ValueError(
            f"k > 1 is only supported for benefit='annuity', got {benefit!r}"
        )

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    pmf = _death_year_pmf(at, x)
    cdf = np.cumsum(pmf)

    u = rng.random(n_sim)
    K = np.searchsorted(cdf, u, side="right")
    K = np.minimum(K, len(pmf) - 1)

    # --- Deferral: alive at m iff K >= m (since T ∈ [K, K+1) and m is integer) ---
    if m > 0:
        alive_at_m = K >= m
        K_eff = np.where(alive_at_m, K - m, 0)  # dummy 0 for dead scenarios
    else:
        alive_at_m = None
        K_eff = K

    # --- k-thly annuity: sample fractional year of death J ~ U{0,..,k-1} ---
    if benefit == "annuity" and k > 1:
        J = rng.integers(0, k, size=n_sim)
        J_eff = J if m == 0 else np.where(alive_at_m, J, 0)
        samples = _pv_annuity_kthly(K_eff, J_eff, at.interest.v, k=k, n=n)
    else:
        samples = _pv_from_death_years(at, K_eff, benefit=benefit, n=n)

    # --- Apply deferral discount and zero out pre-deferral deaths ---
    if m > 0:
        samples = np.where(alive_at_m, at.interest.v ** m * samples, 0.0)

    parts = [f"x={x}", f"n={n}", f"benefit={benefit}"]
    if k != 1:
        parts.append(f"k={k}")
    if m != 0:
        parts.append(f"m={m}")
    parts.append(f"n_sim={n_sim}")
    label = f"simulate_pv({', '.join(parts)})"

    return StochasticResult(np.asarray(samples, dtype=float), label=label)
