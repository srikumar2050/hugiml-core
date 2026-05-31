hugiml-core 1.1.4
===================

``hugiml-core`` provides a C++-accelerated, scikit-learn-compatible implementation of the HUG-IML interpretable rule-based classifier.

Version 1.1.4 keeps the public training behavior while improving the native execution path for two high-impact areas: the L1 fit hot path and adaptive binning.

Highlights
----------

* Native L1 hot path for ``L=1``: fused preparation, single-item mining, top-K retention, and sparse training-matrix construction.
* Native adaptive binning: supervised per-feature bin selection at the C++ layer with Python metadata preserved for inspection and serialization.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   changelog
   api
