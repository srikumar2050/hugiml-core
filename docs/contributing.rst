Contributing
============

The repository includes ``CONTRIBUTING.md`` at the project root. Typical development setup:

.. code-block:: bash

   git clone https://github.com/srikumar2050/hugiml-core.git
   cd hugiml-core
   python -m pip install -e ".[dev,plots,benchmarks]"
   python scripts/build_batched.py --inplace
   python scripts/run_pytest_batches.py

The batched build helper uses conservative defaults for the native C++ extension and avoids the common direct-build failure where ``pybind11`` is missing from the active environment. Use isolated package builds such as ``python -m pip install .`` when building wheels or installs from source; avoid ``--no-build-isolation`` unless the build requirements declared in ``pyproject.toml`` are already installed.

The batched pytest runner writes logs under ``build/pytest_batches/`` and retries timed-out batches one file at a time. Direct ``pytest`` remains available for unconstrained local environments.

Before opening a pull request, run the relevant tests and update documentation when public APIs or examples change.

