# TODO — pylifecontingencies roadmap

## High priority (next sprint)

### 1. DynamicLifeTable — accept external forecasts directly

**Goal:** Let users bring their own mortality-rate predictions from any model
(neural net, gradient boosting, external R/Julia model, etc.) and immediately
build actuarial functions on top of them.

**Design:**

```python
from pylifecontingencies.dynamic import DynamicLifeTable, DynamicActuarialTable

# Option A — cohort table from a single forecast path (ages × years DataFrame)
# User passes predicted qx or mx, we extract the cohort diagonal
dlt = DynamicLifeTable.from_forecast_mx(
    df_mx,             # pd.DataFrame: ages as index, years as columns
    birth_year=1985,
)

# Option B — period table (all ages from same calendar year)
dlt = DynamicLifeTable.from_forecast_qx(
    df_qx,
    period_year=2040,
)

# Option C — multiple scenarios / sample paths (stochastic)
# df_mx_samples: shape (n_samples, n_ages, n_years) or list of DataFrames
dlt = DynamicLifeTable.from_scenarios(df_mx_samples, birth_year=1985)

# Build an actuarial table from it
dat = DynamicActuarialTable(dlt, i=0.03)

# Same API as ActuarialTable — single-path case returns a scalar
dat.axn(x=40, n=25)          # ä_{40:25|}
dat.Axn(x=40)                # A_40 (whole-life)
dat.net_premium(x=40, n=25)  # net premium

# Stochastic case returns a distribution (median + percentiles)
dat.axn(x=40, n=25)
# → { "mean": 14.2, "median": 14.1, "p05": 12.8, "p95": 15.7, "samples": [...] }
```

**Files to create / modify:**

- `src/pylifecontingencies/dynamic/dynamic_lifetable.py` — `DynamicLifeTable` class
- `src/pylifecontingencies/dynamic/dynamic_actuarialtable.py` — `DynamicActuarialTable` class
- `src/pylifecontingencies/dynamic/__init__.py` — re-export new classes
- `tests/test_dynamic_lifetable.py` — unit tests

**Key design decisions:**

- Single-path input → `DynamicLifeTable` wraps a plain `LifeTable` internally;
  `DynamicActuarialTable.axn()` returns a float just like the static API.
- Multi-path (stochastic) input → stores a list of `LifeTable` objects;
  actuarial functions return a `StochasticResult` with `mean`, `std`,
  `quantile(q)`, `samples` array.
- `DynamicActuarialTable` should be a drop-in replacement for `ActuarialTable`
  in the single-path case (same method signatures, same return types).

---

### 2. ProjectedLifeTable improvements

- Accept raw DataFrames directly (not just `LeeCarterForecast` / `CBDForecast`)
  so external forecasts slot in without wrapping.
- `from_mx(df, birth_year)`, `from_qx(df, birth_year)` class methods.
- Handle cohort ages that extend beyond the forecast horizon by extrapolating
  using the last available age pattern (constant log-mortality slope).

---

### 3. Add bundled tables from R lifecontingencies

Run `scripts/convert_rda_to_parquet.py` and commit the generated parquet files:

- `soa08` (2001 CSO)
- `AM92Lt`, `AF92Lt` (UK)
- `demoUsa`, `demoUK`, `demoIta`, `demoFrance`, `demoGermany`, `demoJapan`,
  `demoChina`, `demoCanada`

Validate each against R with `tests/test_actuarial_vs_r.py`.

---

## Medium priority (v2)

### 4. ~~Multi-life actuarial functions~~ ✅ DONE

Implemented in `src/pylifecontingencies/multilife.py` with 40 unit tests
(`tests/test_multilife.py`) + R parity tests (`tests/test_multilife_vs_r.py`).

Demographic:

- `pxyt(ltx, lty, x, y, t, status)` — joint/last-survivor survival probability
- `qxyt(ltx, lty, x, y, t, status)` — joint/last-survivor death probability
- `exyt(ltx, lty, x, y, status)` — joint/last-survivor life expectation

Actuarial EPVs:

- `axyn(atx, aty, x, y, n, k, status)` — joint/last-survivor annuity-due
- `Axyn(atx, aty, x, y, n, k, status)` — joint/last-survivor insurance
- `Exyn(atx, aty, x, y, n, status)` — joint/last-survivor pure endowment
- `AExyn(atx, aty, x, y, n, k, status)` — joint/last-survivor endowment insurance

### 5. Multi-decrement tables (MDT)

- `MultiDecrementTable` class
- `qxt_prime`, `qxt_fromQxprime` — independent/dependent decrements
- `Axn_mdt` — APV of benefit on decrement j

### 6. Stochastic PV simulation (rLifeContingencies equivalent)

- `simulate_pv(at, x, n, benefit, n_sim)` — Monte Carlo PV distribution
- Returns mean, std, full sample vector
- Uses vectorised NumPy random draws over lx probabilities

### 7. Mortality law fitters

- `GompertzMakeham(mu_x = A + B*c^x)` — MLE fit to qx data
- `HeligmanPollard` — 8-parameter model
- `fit_mortality_law(lt, law, ages)` → fitted parameters + goodness-of-fit
- Complement to the `dynamic/` module for parametric graduation

---

## Lower priority (v2+)

### 8. Additional forecasting models (StMoMo family)

- `RenshawHaberman` — LC with cohort effect
- `APC` — Age-Period-Cohort
- `CBD_M6`, `CBD_M7`, `CBD_M8` — extended CBD variants
- `Plat` — Plat (2009) four-factor model
- Shared `MortalityModel` base class with `.fit()` / `.forecast()` / `.simulate()`

### 9. Woolhouse formula for k-thly annuities

Replace the UDD α(m)/β(m) approximation with the three-term Woolhouse formula
for better accuracy at high k (monthly, continuous):

```
ä^(m)_{x:n|} ≈ ä_{x:n|} - (m-1)/(2m)*(1 - _nEx) - (m²-1)/(12m²)*(δ+μ_x)*(1 - _nEx*...)
```

### 10. Continuous functions

- `abar_x`, `Abar_x` — continuous equivalents (δ as force of interest)
- `axn(at, x, k=inf)` — limiting case using `delta` instead of `i^(m)`

### 11. CI/CD and publishing

- GitHub Actions workflow: `pytest` + `pytest --cov` on push
- rpy2 parity job on a matrix that installs R + lifecontingencies
- Publish to PyPI on tag via `python -m build` + `twine upload`
