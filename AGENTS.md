# Repository Guidelines

## Project Structure & Module Organization

This repository uses a `src/` layout. Core package code lives in `src/pylifecontingencies/`, with static actuarial modules such as `lifetable.py`, `actuarial.py`, and `interest.py`, plus forecasting code under `src/pylifecontingencies/dynamic/`. Bundled mortality tables live in `src/pylifecontingencies/data/`. Tests are in `tests/`, and conversion/import utilities are in `scripts/`. Use `notebooks/` for exploration, not production code.

## Build, Test, and Development Commands

Install in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

Run the full test suite:

```bash
pytest
```

Run tests with coverage for package files:

```bash
pytest --cov=src/pylifecontingencies --cov-report=term-missing
```

Run R parity validation only:

```bash
pytest tests/test_actuarial_vs_r.py -v
```

Build a distributable package:

```bash
python -m build
```

## Coding Style & Naming Conventions

Target Python `>=3.10`. Follow existing style: 4-space indentation, clear module-level functions, and descriptive snake_case names for functions, variables, and test fixtures. Preserve the current API naming where actuarial formulas intentionally mirror the R package, including identifiers such as `Axn`, `Exn`, and `IAxn`. Keep new modules focused and colocate related logic under `dynamic/`, `io/`, or top-level package files as appropriate.

## Testing Guidelines

Pytest is the test framework. Add new tests under `tests/` using `test_*.py` filenames and descriptive test names. Prefer deterministic fixtures from `tests/conftest.py` when possible. If a change affects R equivalence, extend the `rpy2` parity tests, but keep them skippable when R or `rpy2` is unavailable.

## Commit & Pull Request Guidelines

Recent history follows short, imperative subjects with conventional prefixes such as `feat:`, `refactor:`, and `docs:`. Keep commits scoped to one logical change. PRs should explain the actuarial or API impact, list test coverage added or updated, and note any dependency on R, `rpy2`, or external mortality data sources. Include example usage when public behavior changes.

## Configuration Notes

Runtime dependencies are NumPy, pandas, SciPy, and statsmodels. `rpy2` is development-only and used for validation, not normal package operation. Avoid committing generated datasets unless they belong in `src/pylifecontingencies/data/`.
