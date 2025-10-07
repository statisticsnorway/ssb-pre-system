# Copilot Instructions for AI Agents

## Project Overview
- **Domain:** Pre-processing system for monthly and quarterly Norwegian National Accounts statistics.
- **Core Components:**
  - `Formula` class (and subclasses): Encapsulates statistical formulas (e.g., Indicator, FDeflate, FInflate, FSum, FSumProd, FMult, FDiv). All formulas ultimately resolve to Indicator instances.
  - `PreSystem` class: Manages a collection of Formula objects and orchestrates their evaluation.
  - `convert` and `convert_step` functions: Convert Pandas DataFrames between frequencies.
- **Source code:** All main logic is in `src/pre_system/`.

## Key Patterns & Conventions
- **Formula Pattern:** All formulas must ultimately be composed of Indicator instances. Use `what` for textual representation and `evaluate` for Pandas-based computation.
- **Data Handling:** All data input/output is via Pandas DataFrames/Series. Test data is in `tests/testdata/` as Parquet files.
- **Testing:**
  - Tests are in `tests/`, named `test_*.py`.
  - Use pytest for running tests: `poetry run pytest` from the project root.
  - Test data is loaded from Parquet files in `tests/testdata/`.
- **Build & Lint:**
  - Use Poetry for dependency management (`pyproject.toml`).
  - Linting/formatting: `black`, `ruff`, and `pre-commit` are configured.
  - Run `pre-commit run --all-files` to check code style.
- **Type hints:**
  - The codebase uses Python type hints for better clarity and error checking.
  - Ensure to include type annotations when adding new functions or methods.
  - Run `poetry run mypy src tests` to check type consistency.
- **Documentation:**
  - Sphinx docs in `docs/`.
  - Build docs with `make html` (Linux/macOS) or `make.bat html` (Windows) from the `docs/` directory.

## Integration & External Dependencies
- **Pandas** is the primary data processing library.
- **Poetry** manages dependencies and packaging.
- **Pre-commit** hooks enforce style and linting.
- **CI/CD:** GitHub Actions workflows for tests and docs.

## Examples
- See `examples/` for usage patterns and data flows.
- Example: To evaluate a formula, instantiate a Formula subclass and call `evaluate` with the required DataFrames.

## Quickstart for AI Agents
- Implement new formulas as subclasses of `Formula` in `src/pre_system/formula.py`.
- Add new system logic in `src/pre_system/pre_system.py`.
- Add or update tests in `tests/` and test data in `tests/testdata/`.
- Follow existing naming and import conventions.

---
For more details, see `README.md` and the [Reference Guide](https://statisticsnorway.github.io/ssb-pre-system/reference.html).
