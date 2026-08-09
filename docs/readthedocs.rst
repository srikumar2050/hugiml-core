Publishing on Read the Docs
===========================

This repository is already configured for Read the Docs. The live page should be rebuilt from the default branch whenever the package version, user guide, examples, or API docstrings change.

Repository files used by Read the Docs
--------------------------------------

* ``.readthedocs.yaml`` at the repository root selects the builder image and points Sphinx to ``docs/conf.py``.
* ``docs/conf.py`` adds ``src`` to ``sys.path`` and mocks optional/native modules so API pages can be imported without compiling the C++ extension on the docs builder.
* ``docs/requirements.txt`` contains documentation-only dependencies.
* ``docs/index.rst`` is the documentation landing page and controls the sidebar order.

Recommended update workflow
---------------------------

#. Commit documentation changes together with the code changes they describe.
#. Make sure ``pyproject.toml`` and ``src/hugiml/__init__.py`` expose the same release version.
#. Run the local Sphinx build before pushing:

   .. code-block:: bash

      python -m pip install -r docs/requirements.txt
      PYTHONPATH=src sphinx-build -b html docs docs/_build/html

#. Push to the default branch used by your Read the Docs project.
#. Open the Read the Docs project dashboard, go to **Builds**, and trigger a build for ``latest`` if the webhook did not start automatically.
#. After the build finishes, open the live documentation and confirm that the footer/title reflects the new version and that the changed pages appear in the sidebar.

Updating the existing ``hugiml-core`` RTD page
----------------------------------------------

Your current public page is ``https://hugiml-core.readthedocs.io/en/latest/``. If it still displays an older version after you push, use this checklist:

#. In Read the Docs, open **Admin → Advanced Settings** and confirm that the default branch matches your GitHub default branch.
#. In **Admin → Automation Rules** or **Admin → Versions**, confirm that ``latest`` tracks the branch you are updating.
#. In **Builds**, start a fresh build for ``latest``. If the previous build is cached, choose the option to wipe the build environment before rebuilding.
#. Check that the build log uses the branch containing the current release documentation.
#. If GitHub pushes are not triggering builds, reconnect the GitHub integration or recreate the webhook from the RTD project admin page.

Local build check
-----------------

Use the same command locally that Read the Docs effectively runs:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   PYTHONPATH=src sphinx-build -b html docs docs/_build/html

For a stricter local quality gate, add ``-W --keep-going``. This treats warnings as errors and is useful before releases:

.. code-block:: bash

   PYTHONPATH=src sphinx-build -b html docs docs/_build/html -W --keep-going

The native ``_hugiml_core`` module is mocked for documentation import purposes. This is intentional: docs builds do not need to compile the extension. Runtime tests and wheel validation should still use the normal package build.

PyPI and docs alignment
-----------------------

Before publishing a release, keep these in sync:

* ``pyproject.toml`` package version.
* ``src/hugiml/__init__.py`` ``__version__``.
* ``CHANGELOG.md`` release notes.
* GitHub release or tag, if you use tags for stable documentation versions.
* Read the Docs ``latest`` build.
* PyPI project URLs and README content.

After a successful Read the Docs build, verify that the PyPI **Documentation** project URL and README badge resolve to the updated page.

Troubleshooting
---------------

* If API pages fail because optional packages are missing, add the module name to ``autodoc_mock_imports`` in ``docs/conf.py`` instead of installing heavy runtime extras on Read the Docs.
* If image links fail, confirm that the image is committed under ``docs/images`` and referenced with a path relative to the ``docs`` directory, such as ``images/example.png``.
* If ``latest`` still shows an older release, confirm that the RTD project tracks the intended branch and that its webhook completed successfully.
* If a page is missing from the sidebar, add it to the appropriate ``.. toctree::`` in ``docs/index.rst``.
