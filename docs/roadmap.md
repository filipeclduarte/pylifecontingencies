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

### 3. ~~Add bundled tables~~ ✅ DONE

**BR-EMS (SUSEP)** — 80 tests in `tests/test_br_ems.py`

12 tables via `scripts/convert_br_ems.py` → CSV in `src/pylifecontingencies/data/`:

- `br_emssb_2021_m/f`, `br_emsmt_2021_m/f` — BR-EMS 2021 (ages 0–117/116)
- `br_emssb_2015_m/f`, `br_emsmt_2015_m/f` — BR-EMS 2015 (ages 0–118)
- `br_emssb_2010_m/f`, `br_emsmt_2010_m/f` — BR-EMS 2010 (ages 0–116/113)

**R lifecontingencies tables** — via `scripts/convert_rda_to_parquet.py` → parquet:

- `soa08` — 2001 CSO (S4 lifetable, ages 0–140)
- `AM92Lt`, `AF92Lt` — UK AM92/AF92 (S4 lifetable)
- `demoUsa`, `demoUk`, `demoIta`, `demoFrance` — lx multi-column data.frames
- `demoGermany`, `demoJapan`, `demoChina`, `demoCanada` — qx multi-column data.frames

All accessible via `load_table(name)`, `load_table(name, column=...)`, `list_tables()`, `list_columns(name)`.

Loader coverage:

- `tests/test_r_tables_loader.py` covers `list_columns()` and `load_table(..., column=...)`
- `tests/test_actuarial_vs_r.py` covers core actuarial parity against R

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

### 6. ~~Stochastic PV simulation (rLifeContingencies equivalent)~~ ✅ DONE

- `simulate_pv(at, x, n, benefit, n_sim, k, m)` — k-thly payments (UDD) and deferral
- Returns `StochasticResult` with `mean`, `std`, `quantile`, `ci`, `summary`, `var`, `tvar`
- `.hist()` — histogram with mean and 95 % CI lines; `.plot()` — empirical CDF
- `.to_dataframe()` — pandas export with column `"pv"`
- Exposed as top-level function and `ActuarialTable.simulate_pv(...)`
- 55 unit tests covering all benefit types, k/m parameters, and risk metrics

### 7. Mortality law fitters — done

- `GompertzMakeham(mu_x = A + B*c^x)` implemented with MLE fit to `q_x`
- `HeligmanPollard` implemented as an 8-parameter graduation model
- `fit_mortality_law(lt, law, ages)` implemented with string or object dispatch
- Returns `MortalityLawFit` with fitted parameters, observed/fitted `q_x`, residuals, and GOF metrics
- Complements the `dynamic/` module for parametric graduation

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

### 11. CI/CD and publishing — done (first version)

- GitHub Actions `ci.yml`: pytest matrix on Python 3.10, 3.11, and 3.12, plus coverage and build checks
- GitHub Actions `r-parity.yml`: installs R + `lifecontingencies` and runs the R parity tests
- GitHub Actions `publish.yml`: runs tests, builds distributions, and publishes to PyPI on `v*` tags via Trusted Publishing
