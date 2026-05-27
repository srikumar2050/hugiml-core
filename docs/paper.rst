Paper and scientific scope
==========================

Reference
---------

Krishnamoorthy, S. (2024). *Interpretable Classifier Models for Decision Support Using High Utility Gain Patterns*. IEEE Access, 12, 126088--126107. DOI: 10.1109/ACCESS.2024.3455563.

Scope reflected in this implementation
--------------------------------------

The package implements the core HUG-IML idea for practical Python use:

* mine High Utility Gain patterns from labelled tabular data;
* transform input data into a pattern-presence representation;
* fit an interpretable classifier over the transformed space;
* expose pattern-level explanation data suitable for review;
* support benchmark, calibration, governance, monitoring, and deployment workflows.

The documentation emphasizes decision-support contexts such as finance, healthcare, insurance, program integrity, and risk analytics because these are settings where transparent model behavior and audit artifacts are often as important as raw predictive score.

Citation
--------

.. code-block:: bibtex

   @article{krishnamoorthy2024hugIML,
     author  = {Krishnamoorthy, Srikumar},
     title   = {Interpretable Classifier Models for Decision Support Using High Utility Gain Patterns},
     journal = {IEEE Access},
     volume  = {12},
     pages   = {126088--126107},
     year    = {2024},
     doi     = {10.1109/ACCESS.2024.3455563}
   }

