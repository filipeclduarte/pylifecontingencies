# TODO — pylifecontingencies roadmap

## High priority (next sprint)

### 1. ~~DynamicLifeTable — accept external forecasts directly~~ ✅ DONE

Implemented in `src/pylifecontingencies/dynamic/` with 31 unit tests
(`tests/test_dynamic_lifetable.py`).

**Files created:**

- `dynamic_lifetable.py` — `DynamicLifeTable` class with constructors:
  `from_forecast_mx`, `from_forecast_qx`, `from_forecast_log_mx`,
  `from_scenarios`, `from_scenarios_array`
- `dynamic_actuarialtable.py` — `DynamicActuarialTable` (wraps DynamicLifeTable + interest rate);
  single-path returns float, stochastic returns `StochasticResult`
- `stochastic.py` — `StochasticResult` container with `mean`, `std`, `quantile(q)`, `ci(level)`
- `dynamic/__init__.py` — re-exports new classes

---

### 2. ~~ProjectedLifeTable improvements~~ ✅ DONE

Consolidated `ProjectedLifeTable` and `DynamicLifeTable` into a single
model-agnostic class (`projected_table.py`). 29 tests
(`tests/test_projected_table.py`). Backward-compatible with model forecast
objects and `DynamicLifeTable`.

**Class methods** (accept raw DataFrames from any model):

- `ProjectedLifeTable.from_mx(df, lower=..., upper=..., birth_year=...)`
- `ProjectedLifeTable.from_qx(df, lower=..., upper=..., birth_year=...)`
- `ProjectedLifeTable.from_log_mx(df, lower=..., upper=..., birth_year=...)`
- `ProjectedLifeTable.from_scenarios(list_of_dfs, birth_year=...)`
- `ProjectedLifeTable.from_scenarios_array(arr, ages=..., years=..., birth_year=...)`

**Prediction intervals**: pass `lower` and `upper` DataFrames to any constructor.
Produces `.lower`, `.lifetable` (central), `.upper` properties.

**Extrapolation strategies**: `"clamp"` (default), `"constant_slope"`, `"none"`.

**Integration**: `DynamicActuarialTable` now accepts both `ProjectedLifeTable`
and `DynamicLifeTable`. PI/stochastic tables return `StochasticResult`.

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
