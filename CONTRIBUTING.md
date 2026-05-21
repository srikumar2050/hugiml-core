# Contributing to hugiml-core

Thank you for your interest in contributing to **hugiml-core** — the high-performance
interpretable rule-based ML infrastructure library built on the
[HUG-IML IEEE Access paper](https://doi.org/10.1109/ACCESS.2024.3455563).

---

## Before You Start

This project is maintained by **Srikumar Krishnamoorthy** ([@srikumar2050](https://github.com/srikumar2050)).
All contributions — code, documentation, tests, benchmarks — are welcome and reviewed
on a best-effort basis.

By submitting a pull request you agree to the terms of the
[Developer Certificate of Origin (DCO)](https://developercertificate.org/) and that
your contribution will be licensed under the
[Apache License 2.0](LICENSE).

---

## Developer Certificate of Origin (DCO)

Every commit must be signed off with:

```
Signed-off-by: Your Name <your@email.com>
```

Add `-s` to your `git commit` command, or configure Git globally:

```bash
git config --global user.name  "Your Name"
git config --global user.email "your@email.com"
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/srikumar2050/hugiml-core.git
cd hugiml-core

# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install in editable mode with all dev extras
pip install -e ".[dev]"

# Build the C++ extension (requires a C++17 compiler)
python setup.py build_ext --inplace
```

---

## Running the Test Suite

```bash
# Fast unit tests (no native extension required for pure-Python paths)
pytest tests/ -v -m "not integration and not stress"

# Full suite including real-dataset integration tests
pytest tests/ -v

# Stress tests (memory, concurrency)
pytest tests/test_stress.py -v -m stress
```

Coverage must stay at or above **80%**. Check locally with:

```bash
pytest --cov=hugiml --cov-report=term-missing tests/
```

---

## Code Style

```bash
# Linting
ruff check src/ tests/ benchmarks/

# Type checking
mypy src/hugiml/

# Auto-format
ruff format src/ tests/ benchmarks/
```

All checks run automatically in CI on every pull request.

---

## Pull Request Guidelines

1. **Fork** the repository and create a branch from `main`.
2. Write tests that cover your change. All new behaviour should have at least
   one test.
3. Keep pull requests focused — one logical change per PR.
4. Update docstrings and `docs/` if you change public API.
5. Ensure CI passes before requesting a review.
6. Do not bump version numbers in PRs; releases are managed by the maintainer.

---

## Reporting Issues

Please open a [GitHub Issue](https://github.com/srikumar2050/hugiml-core/issues)
and include:

- Python version and OS
- hugiml-core version (`python -c "import hugiml; print(hugiml.__version__)"`)
- Minimal reproducible example
- Full traceback

---

## Security Issues

Do **not** open a public issue for security vulnerabilities.
Email the maintainer directly (see the GitHub profile) with subject
`[hugiml-core] Security`.

---

## License

All contributions must be compatible with [Apache 2.0](LICENSE).
Do not introduce code, data, or dependencies that carry GPL, AGPL,
LGPL, CC-NC, or similarly restrictive terms.
