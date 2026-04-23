"""
pylifecontingencies — life contingencies and actuarial mathematics in Python.

A native Python port of the R `lifecontingencies` package with dynamic
life tables and mortality forecasting.
"""

from .interest import (
    InterestRate,
    convertible2Effective,
    effective2Convertible,
    discount2Interest,
    interest2Discount,
    intensity2Interest,
    interest2Intensity,
    nominal2Real,
    real2Nominal,
)
from .fractional import FractionalAge
from .lifetable import LifeTable
from .actuarialtable import ActuarialTable
from .demographic import (
    pxt,
    qxt,
    dxt,
    mxt,
    Lxt,
    Tx,
    exn,
    ex_complete,
    mx2qx,
    qx2mx,
    getOmega,
    probs2lifetable,
)
from .actuarial import (
    Exn,
    axn,
    Axn,
    AExn,
    IAxn,
    DAxn,
)
# exn is also exported from actuarial (it wraps demographic.exn via ActuarialTable)
from .actuarial import exn as _exn_at  # noqa: F401 — internal, not re-exported

from .premiums import net_premium, gross_premium
from .reserves import prospective_reserve, retrospective_reserve, reserve_recursion
from .financial import (
    presentValue,
    accumulatedValue,
    annuity,
    increasingAnnuity,
    decreasingAnnuity,
    duration,
    convexity,
)
from .simulation import simulate_pv
from .mortalitylaws import GompertzMakeham, HeligmanPollard, MortalityLawFit, fit_mortality_law
from .data import load_table, list_tables
from .dynamic import DynamicLifeTable, DynamicActuarialTable, StochasticResult
from .multilife import (
    pxyt,
    qxyt,
    exyt,
    axyn,
    Axyn as Axyn_ml,  # avoid clash with single-life Axn
    Exyn,
    AExyn,
)

__all__ = [
    # Interest
    "InterestRate",
    "convertible2Effective",
    "effective2Convertible",
    "discount2Interest",
    "interest2Discount",
    "intensity2Interest",
    "interest2Intensity",
    "nominal2Real",
    "real2Nominal",
    # Life table
    "FractionalAge",
    "LifeTable",
    "ActuarialTable",
    # Demographic
    "pxt",
    "qxt",
    "dxt",
    "mxt",
    "Lxt",
    "Tx",
    "exn",
    "ex_complete",
    "mx2qx",
    "qx2mx",
    "getOmega",
    "probs2lifetable",
    # Actuarial values
    "Exn",
    "axn",
    "Axn",
    "AExn",
    "IAxn",
    "DAxn",
    # Premiums & reserves
    "net_premium",
    "gross_premium",
    "prospective_reserve",
    "retrospective_reserve",
    "reserve_recursion",
    # Financial
    "presentValue",
    "accumulatedValue",
    "annuity",
    "increasingAnnuity",
    "decreasingAnnuity",
    "duration",
    "convexity",
    "simulate_pv",
    # Mortality laws
    "GompertzMakeham",
    "HeligmanPollard",
    "MortalityLawFit",
    "fit_mortality_law",
    # Data
    "load_table",
    "list_tables",
    # Dynamic
    "DynamicLifeTable",
    "DynamicActuarialTable",
    "StochasticResult",
    # Multi-life
    "pxyt",
    "qxyt",
    "exyt",
    "axyn",
    "Axyn_ml",
    "Exyn",
    "AExyn",
]
