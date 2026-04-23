# Release Checklist

## Version choice

- If this is the first public release on PyPI, keep `0.1.0`.
- If `0.1.0` was already tagged or published anywhere, bump to `0.1.1`.
- Update `version` in `pyproject.toml` before tagging.

## Pre-release checks

- Run `pytest`
- Run `pytest --cov=src/pylifecontingencies --cov-report=term-missing`
- Run `pytest tests/test_actuarial_vs_r.py tests/test_multilife_vs_r.py -v` if R + `lifecontingencies` are available
- Run `python -m build`
- Run `python -m twine check dist/*`

## Packaging checks

- Confirm bundled data files are present in the built artifacts:
  `soa_ilt.csv`, BR-EMS CSVs, and R-sourced `.parquet` tables
- Confirm `README.md` renders correctly on PyPI
- Confirm runtime dependencies in `pyproject.toml` are correct, especially `pyarrow`

## GitHub / PyPI setup

- Push the current branch
- Confirm GitHub Actions workflows are green:
  `CI`, `R Parity`, and `Publish`
- Configure PyPI Trusted Publishing for the repository, or confirm the existing setup
- Confirm the `pypi` GitHub environment is available if `publish.yml` uses it

## Release steps

- Commit final release changes
- Create the tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

- Watch the `Publish` GitHub Actions workflow
- Confirm the new version appears on PyPI

## Post-release checks

- Run `pip install pylifecontingencies` in a clean environment
- Import the package and smoke-test:

```python
from pylifecontingencies import load_table, ActuarialTable, axn
lt = load_table("soa_ilt")
at = ActuarialTable(lt, 0.03)
print(axn(at, x=40))
```

- Verify at least one parquet-backed table loads:

```python
from pylifecontingencies import load_table
lt = load_table("soa08")
print(lt)
```
