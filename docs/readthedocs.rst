Publishing on Read the Docs
===========================

This package includes a ready-to-use Read the Docs configuration:

* ``.readthedocs.yaml`` at the repository root.
* ``docs/conf.py`` for the Sphinx build.
* ``docs/requirements.txt`` for documentation dependencies.

Recommended setup
-----------------

#. Push the repository to GitHub.
#. Sign in to Read the Docs and import the GitHub repository.
#. Confirm that the project slug is ``hugiml-core`` if you want the documentation URL to match ``https://hugiml-core.readthedocs.io``.
#. In Read the Docs project settings, enable builds for the default branch and release tags.
#. Trigger a build.

Local build check
-----------------

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   PYTHONPATH=src sphinx-build -b html docs docs/_build/html

The Sphinx config adds ``src`` to ``sys.path`` and mocks the compiled ``_hugiml_core`` extension for API-documentation import purposes. Read the Docs therefore does not need to compile the native extension just to publish documentation. Local runtime testing still requires the normal package build or an installed wheel.

PyPI and docs alignment
-----------------------

The package metadata already points to the expected documentation URL through the ``Documentation`` project URL in ``pyproject.toml``. After the first successful Read the Docs build, confirm that the badge and PyPI project link resolve correctly.

