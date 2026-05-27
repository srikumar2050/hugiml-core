Contributing
============

The repository includes ``CONTRIBUTING.md`` at the project root. Typical development setup:

.. code-block:: bash

   git clone https://github.com/srikumar2050/hugiml-core.git
   cd hugiml-core
   python -m pip install -e ".[dev,plots,benchmarks]"
   python setup.py build_ext --inplace
   pytest

Before opening a pull request, run the relevant tests and update documentation when public APIs or examples change.

