# pylifecontingencies — developer guide for Claude

## Project summary

Native Python port of the R `lifecontingencies` package, plus a `dynamic/` module with Lee-Carter and CBD M5 mortality forecasting. No R dependency at runtime; rpy2 is only used in the test suite for numerical parity validation.

Working directory: `/Users/filipeduarte/pylifecontingencies`

## Install (editable)

```bash
pip install -e ".[dev]"
```

Requires R + lifecontingencies CRAN package for rpy2-backed parity tests:

```bash
# macOS
brew install r
Rscript -e "install.packages('lifecontingencies', repos='https://cloud.r-project.org')"
```

## Build and test commands

```bash
# Run all tests (skips rpy2 tests if R not installed)
pytest

# Run with coverage
pytest --cov --cov-report=term-missing

# Run only rpy2 parity tests (requires R)
pytest tests/test_actuarial_vs_r.py -v

# Run static analysis
python -m py_compile src/pylifecontingencies/**/*.py

# Build wheel
pip install build && python -m build
```

## Package layout

```
src/pylifecontingencies/
├── interest.py          # InterestRate: i ↔ d ↔ δ, nominal↔effective, i^(m), d^(m)
├── fractional.py        # FractionalAge enum: UDD, CONST_FORCE, BALDUCCI
├── lifetable.py         # LifeTable: from_qx / from_lx / from_mx + npx / nqx
├── actuarialtable.py    # ActuarialTable = LifeTable + InterestRate + commutation Dx/Nx/Cx/Mx/Rx
├── demographic.py       # pxt, qxt, dxt, mxt, Lxt, Tx, exn, mx2qx, qx2mx, getOmega
├── actuarial.py         # axn, Axn, Exn, IAxn, DAxn, AExn  (single-life EPVs)
├── multilife.py         # pxyt, qxyt, exyt, axyn, Axyn, Exyn, AExyn  (two-life joint/last-survivor)
├── premiums.py          # net_premium, gross_premium
├── reserves.py          # prospective_reserve, retrospective_reserve
├── financial.py         # annuity, presentValue, accumulatedValue, duration, convexity
├── data/
│   ├── __init__.py      # load_table(name) -> LifeTable
│   └── soa_ilt.csv      # SOA Illustrative Life Table (ages 0–99)
├── io/
│   ├── hmd.py           # HMD flat-file loader (requires HMD account credentials)
│   └── soa.py           # SOA XTbML via pymort (optional extra)
└── dynamic/
    ├── rates.py          # MortalityRates: 2-D (age × year) DataFrame wrapper
    ├── leecarter.py      # LeeCarter: SVD fit + ARIMA kt forecast + bootstrap PI
    ├── cbd.py            # CBD M5: per-year logistic OLS + bivariate RW forecast
    ├── bootstrap.py      # residual_bootstrap, parametric_bootstrap helpers
    └── projected_table.py# ProjectedLifeTable: from_mx/from_qx/from_log_mx + cohort/period + extrapolation
```

## Key design decisions

- **LifeTable** stores `lx` as a numpy float64 array from `x_min` to `omega` (inclusive, so `l_omega = 0`). Index math: `lx[k]` = survivors at age `x_min + k`.
- **ActuarialTable** computes commutation columns `Dx, Nx, Cx, Mx, Rx` as cached properties using vectorised NumPy. All EPV functions delegate to these.
- **axn / Axn / ...** are module-level functions that accept an `ActuarialTable` as first argument, matching R's interface exactly. They also work as methods via `ActuarialTable.axn(x, n)`.
- **Fractional-age** logic lives in `FractionalAge` and is stored on `LifeTable`. Affects `npx(x, n)` for non-integer `n`. Default is UDD.
- **k-thly** payments (k>1) use the UDD approximation: `ä^(m) = α(m)ä - β(m)(1 - _nEx)` for annuities; `A^(m) = (i/i^(m)) A` for insurances.
- **Dynamic module** is fully decoupled from the static module — `MortalityRates` is a plain DataFrame wrapper, and `ProjectedLifeTable.to_life_table()` returns a plain `LifeTable`.

## Commutation columns (reference)

```
Dx  = v^x * lx
Nx  = sum_{t=x}^{omega-1} Dt     (annuity numerator)
Cx  = v^{x+1} * dx
Mx  = sum_{t=x}^{omega-1} Ct     (insurance numerator)
Rx  = sum_{t=x}^{omega-1} Mt     (increasing insurance numerator)
```

## Core EPV formulas (reference)

```
_n E_x          = Dx+n / Dx
ä_{x:n|}        = (Nx - Nx+n) / Dx
A^1_{x:n|}      = (Mx - Mx+n) / Dx
A_{x:n|}        = (Mx - Mx+n + Dx+n) / Dx     [endowment]
(IA)_x          = Rx / Dx
(IA)^1_{x:n|}   = (Rx - Rx+n - n*Mx+n) / Dx
(DA)^1_{x:n|}   = ((n+1)(Mx-Mx+n) - (Rx-Rx+n-n*Mx+n)) / Dx
e_x (curtate)   = (Nx+1) / Dx
```

## Multi-life EPV formulas (reference)

Assumes independent future lifetimes. `status` = `"joint"` (both alive) or `"last"` (at least one alive).

```
Joint-life:
  ₜp_{xy}             = ₜpₓ · ₜp_y
  ä_{xy:n|}           = Σ_{t=0}^{n-1} vᵗ · ₜp_{xy}
  A_{xy:n|}           = Σ_{t=0}^{n-1} v^{t+1} · ₜp_{xy} · (1 - p_{x+t}·p_{y+t})
  ₙE_{xy}             = vⁿ · ₙp_{xy}
  e_{xy}              = Σ_{t≥1} ₜp_{xy}

Last-survivor (inclusion-exclusion):
  ₜp_{x̄ȳ}            = ₜpₓ + ₜp_y - ₜp_{xy}
  ä_{x̄ȳ}             = äₓ + ä_y - ä_{xy}
  A_{x̄ȳ}             = Aₓ + A_y - A_{xy}
  e_{x̄ȳ}             = eₓ + e_y - e_{xy}

Identity:  A_{xy} + d · ä_{xy} = 1
```

## Bundled data

`data/soa_ilt.csv` — SOA Illustrative Life Table from Bowers et al. "Actuarial Mathematics" (2nd ed., Appendix 2A). Ages 0–99, omega = 100.

To import additional tables from the R lifecontingencies package (AM92, AF92, soa08, demoUsa, etc.), run:

```bash
python scripts/convert_rda_to_parquet.py
```

This requires `rpy2` and R with `lifecontingencies` installed. Outputs go to `src/pylifecontingencies/data/`.

## Validation strategy

`tests/test_actuarial_vs_r.py` uses `rpy2` to call R's `lifecontingencies` and asserts Python values match to `atol=1e-10`. Tests are automatically skipped if `rpy2` or R is not installed. To run them:

```bash
pytest tests/test_actuarial_vs_r.py -v -s
```

The grid covers: `soa_ilt` × ages [20, 40, 60] × terms [1, 10, 20, Inf] × i [0.01, 0.03, 0.06].

## Forecasting notes

**Lee-Carter (leecarter.py):**

- SVD on (log(mx) - row-means) to extract ax, bx, kt
- Convention: bx normalised so sum(bx) = 1, and kt re-centred so sum(kt) = 0 (first stage)
- kt forecast via `statsmodels.tsa.arima.model.ARIMA(kt, order=(0,1,0))` with drift by default
- Bootstrap: resample Pearson residuals `(log(mx) - (ax + bx * kt)) / sigma_hat`

**CBD M5 (cbd.py):**

- Per-year OLS of `logit(qx_t)` on `[1, (x - x_bar)]` gives `(k1_t, k2_t)`
- x_bar = mean age over the calibration range
- Forecast: bivariate random walk with drift fitted via OLS on lagged differences
