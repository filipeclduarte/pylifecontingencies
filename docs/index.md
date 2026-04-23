# pylifecontingencies

A native Python port of the R [`lifecontingencies`](https://github.com/spedygiorgio/lifecontingencies) package, extended with **dynamic life tables and mortality-rate forecasting** (Lee-Carter, CBD M5).

No R runtime required. Pure NumPy + pandas at install time; `rpy2` is only used in the validation test suite.

---

## What it includes

- Single-life actuarial present values: annuities, insurances, endowments, increasing/decreasing benefits, premiums, and reserves
- Mortality and demographic utilities: `pxt`, `qxt`, `mxt`, `Lxt`, `Tx`, `exn`
- Dynamic mortality forecasting: Lee-Carter, CBD M5, projected life tables, stochastic scenario support
- Monte Carlo PV simulation via `StochasticResult` — k-thly payments, deferral, pandas export, visualisation (`hist`, `plot`), and risk metrics (`var`, `tvar`)
- Parametric mortality graduation with `GompertzMakeham` and `HeligmanPollard`
- Bundled data: SOA ILT, BR-EMS tables, and R-sourced parquet tables such as `soa08`, `AM92Lt`, `AF92Lt`, `demoUsa`, and `demoGermany`

## Installation

Core package:

```bash
pip install pylifecontingencies
```

With SOA XTbML support:

```bash
pip install "pylifecontingencies[soa]"
```

For local development and release checks:

```bash
pip install -e ".[dev]"
```

---

## Quick examples

### Static actuarial values

```python
from pylifecontingencies import load_table, ActuarialTable, axn, Axn, AExn

lt = load_table("soa_ilt")
at = ActuarialTable(lt, interest=0.06)

axn(at, x=40)          # whole-life annuity-due
Axn(at, x=40, n=20)    # 20-year term insurance
AExn(at, x=40, n=20)   # 20-year endowment insurance
```

### Bundled R tables

```python
from pylifecontingencies import load_table, list_columns

list_columns("demoUsa")
lt_usa = load_table("demoUsa", column="USSS2007M")
lt_soa = load_table("soa08")
```

### BR-EMS tables

```python
from pylifecontingencies import load_table, ActuarialTable, axn, qxt

# BR-EMS Survival 2021, male
lt_br = load_table("br_emssb_2021_m")
at_br = ActuarialTable(lt_br, interest=0.03)

qxt(lt_br, x=40, t=10)   # 10-year death probability at age 40
axn(at_br, x=65)         # whole-life annuity-due at age 65
```

### Stochastic simulation

```python
from pylifecontingencies import simulate_pv

# Annual term insurance
r = simulate_pv(at, x=40, n=20, benefit="term", n_sim=50_000, random_state=42)
r.mean, r.std, r.quantile(0.95)

# Monthly annuity (k=12)
r_monthly = simulate_pv(at, x=40, n=20, benefit="annuity", k=12, n_sim=50_000)

# 10-year deferred whole-life annuity
r_deferred = simulate_pv(at, x=40, benefit="annuity", m=10, n_sim=50_000)

# Visualisation
r.hist()          # histogram with mean and 95 % CI lines
r.plot()          # empirical CDF

# Risk metrics (Solvency II)
r.var(0.995)      # Value at Risk at 99.5 %
r.tvar(0.995)     # Tail VaR (Expected Shortfall) at 99.5 %

# Export samples to pandas
df = r.to_dataframe()   # DataFrame with column "pv"
```

### Mortality-law fitting

```python
import numpy as np
from pylifecontingencies import fit_mortality_law

fit = fit_mortality_law(lt, "gompertz_makeham", ages=np.arange(40, 90))
fit.params_dict, fit.rmse
```

---

## Development and release

GitHub Actions workflows live under `.github/workflows/`:

- `ci.yml` runs the main pytest matrix on Python 3.10, 3.11, and 3.12, plus coverage and distribution builds.
- `r-parity.yml` installs R + `lifecontingencies` and runs the parity tests that compare Python results against the R package.
- `publish.yml` builds and publishes to PyPI on tags matching `v*` after the test suite passes.

For publishing, configure GitHub Trusted Publishing for the PyPI project, then create a version tag such as:

```bash
git tag v0.1.0
git push origin v0.1.0
```

---

## Quick start — static life table

```python
from pylifecontingencies import load_table, ActuarialTable
from pylifecontingencies import axn, Axn, Exn, AExn, IAxn, DAxn, exn

# Load a bundled table
lt = load_table("soa_ilt")            # SOA Illustrative Life Table
at = ActuarialTable(lt, interest=0.06)

# Whole-life annuity-due at age 40
axn(at, x=40)                         # ä_40

# 20-year term insurance
Axn(at, x=40, n=20)                   # A^1_{40:20|}

# 20-year endowment insurance
AExn(at, x=40, n=20)                  # A_{40:20|}

# 20-year pure endowment
Exn(at, x=40, n=20)                   # _20 E_40

# Increasing whole-life insurance
IAxn(at, x=40)                        # (IA)_40

# 20-year decreasing term
DAxn(at, x=40, n=20)                  # (DA)^1_{40:20|}

# Curtate future lifetime expectation
exn(at, x=40)                         # e_40
```

### Semi-annual payments (k=2)

```python
axn(at, x=40, n=20, k=2)             # ä^(2)_{40:20|} via UDD
Axn(at, x=40, n=20, k=2)             # A^(2)_{40:20|} via UDD
```

---

## Quick start — dynamic life tables

```python
from pylifecontingencies.dynamic import MortalityRates, LeeCarter, ProjectedLifeTable

# Build a rates surface from a DataFrame (ages as index, years as columns, values = log(mx))
rates = MortalityRates.from_dataframe(df_log_mx)

# Fit Lee-Carter
lc = LeeCarter().fit(rates)
print(lc.ax)   # age-specific levels
print(lc.bx)   # age-specific sensitivities
print(lc.kt)   # period index

# Forecast 50 years ahead with 95% bootstrap prediction interval
forecast = lc.forecast(horizon=50, n_bootstrap=500, ci=0.95)

# Build a cohort life table for someone born in 1985
cohort_lt = ProjectedLifeTable(forecast, birth_year=1985).to_life_table()
at_cohort = ActuarialTable(cohort_lt, interest=0.03)
axn(at_cohort, x=40)   # cohort-true annuity at 40
```

---

## Stochastic PV simulation

```python
from pylifecontingencies import load_table, ActuarialTable, simulate_pv

lt = load_table("soa_ilt")
at = ActuarialTable(lt, interest=0.03)

# Annual term insurance
r = simulate_pv(at, x=40, n=20, benefit="term", n_sim=50_000, random_state=42)
print(r.mean, r.std)
print(r.quantile(0.95))

# Monthly annuity (k=12 payments per year, UDD approximation)
r_monthly = simulate_pv(at, x=40, n=20, benefit="annuity", k=12, n_sim=50_000)

# 10-year deferred whole-life annuity
r_deferred = simulate_pv(at, x=40, benefit="annuity", m=10, n_sim=50_000)

# Monthly pension starting in 10 years (k and m combined)
r_pension = simulate_pv(at, x=40, n=20, benefit="annuity", k=12, m=10, n_sim=50_000)

# Visualisation
r.hist()           # histogram with mean and 95 % CI lines
r.plot()           # empirical CDF (step-wise)

# Risk metrics (Solvency II / reserving)
r.var(0.995)       # Value at Risk at 99.5 %
r.tvar(0.995)      # Tail VaR (Expected Shortfall) at 99.5 %

# Export samples to pandas
df = r.to_dataframe()   # DataFrame with column "pv"
df["pv"].describe()

# Same API as a method on the table
r_ann = at.simulate_pv(x=40, n=20, benefit="annuity", k=12, n_sim=10_000)
```

Supported benefit types: `term`, `whole` / `whole_life`, `annuity`,
`pure_endowment`, `endowment`, `increasing`, `decreasing`.

`k > 1` (fractional payments, UDD) and `m > 0` (deferral) are supported for `annuity`.
`m > 0` is also supported for all insurance benefit types.

---

## Mortality-law graduation

```python
import numpy as np
from pylifecontingencies import (
    load_table,
    fit_mortality_law,
    GompertzMakeham,
    HeligmanPollard,
)

lt = load_table("soa_ilt")

# String dispatch
gm_fit = fit_mortality_law(lt, "gompertz_makeham", ages=np.arange(40, 90))
print(gm_fit.params_dict)
print(gm_fit.rmse, gm_fit.aic)

# Explicit law object
hp_fit = fit_mortality_law(lt, HeligmanPollard(), ages=np.arange(1, 90))
df_fit = hp_fit.to_dataframe()
```

`MortalityLawFit` stores fitted parameters, observed/fitted `q_x`, residuals,
and goodness-of-fit metrics (`loglik`, `AIC`, `BIC`, `RMSE`, `MAE`).

---

## Interest-rate utilities

```python
from pylifecontingencies import InterestRate

ir = InterestRate(i=0.05)
ir.v           # 0.952...  discount factor
ir.delta       # 0.04879...  force of interest
ir.d           # 0.04762...  annual discount rate
ir.i_m(12)     # monthly nominal rate
ir.d_m(12)     # monthly nominal discount rate

InterestRate.from_delta(0.05)    # from force of interest
InterestRate.from_discount(0.04) # from annual discount rate
```

---

## Demographic functions

```python
from pylifecontingencies import pxt, qxt, dxt, mxt, Lxt, Tx

lt = load_table("soa_ilt")
pxt(lt, x=40, t=10)   # _10 p_40
qxt(lt, x=40, t=10)   # _10 q_40
exn(lt, x=40)         # e_40  (curtate)
```

---

## Bundled tables

| Name | Description |
|------|-------------|
| `soa_ilt` | SOA Illustrative Life Table (Bowers et al., ages 0–99) |

R-sourced parquet tables are also bundled, including `soa08`, `AM92Lt`,
`AF92Lt`, `demoUsa`, `demoUk`, `demoFrance`, `demoIta`, `demoGermany`,
`demoJapan`, `demoChina`, and `demoCanada`.

### Multi-column R tables

Some bundled parquet datasets contain several sub-tables in one file. Use
`list_columns(name)` to inspect the available columns and `load_table(..., column=...)`
to select one:

```python
from pylifecontingencies import load_table, list_columns

list_columns("demoUsa")
# ['USSS2007M', 'USSS2007F', 'USSS2000M', 'USSS2000F', 'USSS1990M', 'USSS1990F']

lt_usa = load_table("demoUsa", column="USSS2007M")
lt_ger = load_table("demoGermany", column="qxMale")
```

Single-column parquet tables such as `soa08`, `AM92Lt`, and `AF92Lt` can be loaded
directly:

```python
lt = load_table("soa08")
```

Additional tables (AM92, AF92, demoUsa, etc.) can be imported from R using the provided conversion script:

```bash
# requires R + lifecontingencies + rpy2 + pyarrow
python scripts/convert_rda_to_parquet.py
```

---

## Comparison with R lifecontingencies

| R | Python |
|---|--------|
| `axn(at, x=40, n=20)` | `axn(at, x=40, n=20)` |
| `Axn(at, x=40, n=20)` | `Axn(at, x=40, n=20)` |
| `Exn(at, x=40, n=20)` | `Exn(at, x=40, n=20)` |
| `exn(lt, x=40)` | `exn(lt, x=40)` |
| `pxt(lt, x=40, t=5)` | `pxt(lt, x=40, t=5)` |
| `new("lifetable", x=..., lx=..., name=...)` | `LifeTable(lx, x_min=0, name=...)` |
| `new("actuarialtable", ..., interest=0.06)` | `ActuarialTable(lt, interest=0.06)` |

---

## Scope and roadmap

**Current:** Single-life EPVs, stochastic PV simulation, mortality-law fitters, interest-rate utilities, demographic functions, bundled tables, Lee-Carter and CBD M5 mortality forecasting.

**Planned next:** Multi-decrement tables, Renshaw-Haberman and APC forecasting models, and broader stochastic simulation equivalents to `rLifeContingencies`.

## License

GPL-2.0 — matching the upstream `lifecontingencies` R package.
