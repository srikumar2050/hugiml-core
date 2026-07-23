RPTE downstream model
=====================

Overview
--------

The Residual Pattern Tree Ensemble (RPTE) is an optional downstream model for
HUGIML. HUGIML first creates an interpretable supplied-feature representation
from original features, mined patterns, and configured pair features. Original
features, generated pairs, and order-one/two patterns may enter RPTE trees, while
patterns above order two remain direct sparse terms. A sparse L1 logistic
layer is then fitted over accepted terminal leaves and remaining direct terms.

RPTE is intended for cases where a linear combination of HUGIML terms is too
restrictive, particularly when predictive evidence appears only through
higher-order relationships. It remains intrinsically inspectable: the fitted
model is represented by explicit root-to-leaf conjunctions and explicit direct
terms rather than by a separate post-hoc explanation model.

Conceptual model
----------------

RPTE separates representation learning into three interpretable stages:

#. **Supplied HUGIML terms.** Original features, mined patterns, and configured
   pair features form the candidate representation. Patterns of order one or
   two may participate in tree growth; higher-order patterns remain direct
   sparse terms.
#. **Residual pattern trees.** Shallow trees are added sequentially against the
   current classification residuals. A tree captures a bounded conjunction of
   supplied terms, and every accepted terminal leaf becomes an explicit binary
   rule feature.
#. **Sparse final combination.** An L1 logistic layer combines the accepted leaf
   rules with supplied terms that were not used by a tree. When a leaf and an
   unused mined pattern describe the same atom conjunction, raw-feature
   ownership, and training support, RPTE retains the leaf as the canonical term
   and records the pattern as an alias.

This structure makes the prediction path reviewable at both global and
instance levels. Reviewers can inspect which direct terms remain, which leaf
rules were accepted, how each rule was formed, its support, its coefficient,
and which rules activate for a particular row. The alias table preserves
representation provenance without inflating model or inspection-unit counts.

How tree growth works
---------------------

``enable_lookahead=False`` uses sequential shallow trees. Each round grows a
standard tree against the current boosting residuals.

``enable_lookahead=True`` uses the native bounded-lookahead grower. At each
eligible leaf it first evaluates an ordinary split. When that split has
insufficient held-out gain, RPTE can evaluate a compact root-and-child
microtree. The root may use an augmented pair supplied by HUGIML or a raw pair synthesized
inside the native search, and the first child may use a single source or another
pair. Configured deeper leaf structures can extend this bounded search one
additional level for higher-order relationships. Mined pair patterns are root
candidates when the configured ``lookahead_ops`` admits
``"pattern"``; patterns above order two are excluded from all tree positions.

This is interaction relaxation through the first child split: weak marginal
features are not admitted globally. They are considered only inside a bounded
joint structure, and the structure is retained only when it improves held-out
probe gain and passes the configured statistical acceptance checks.

``enable_lookahead="adaptive"`` chooses between the sequential and lookahead
paths from the fitted data. It uses lookahead when single-source marginal signal
is weak enough that bounded joint search is likely to add value.

Boosting and final sparse layer
-------------------------------

Each accepted tree is fitted to the current binomial-deviance residuals. RPTE
computes Newton leaf values and applies a backtracking step, retaining a tree
only when the empirical deviance decreases. The final L1 logistic estimator receives:

* one indicator for every accepted terminal leaf; and
* every supplied downstream column that was not selected in an accepted tree
  split, subject to structural alias canonicalization.

Mining depth ``L`` above two does not expand RPTE's tree grammar. Singleton and
pair patterns retain their existing eligibility, while patterns of order three
or greater are direct-only. Original columns and augmented or synthesized pair
columns remain eligible for tree growth.

Before fitting the final logistic layer, RPTE compares unused mined pattern
indicators with terminal leaves. A direct pattern is omitted only when the leaf
represents the same positive atom conjunction, both representations have the
same raw-feature ownership, and their nonconstant binary support is identical on
the fitting data. The retained leaf is the canonical term,
and ``representation_alias_table()`` records the omitted pattern, matching leaf,
raw sources, and support. This prevents L1 coefficient allocation and complexity
counts from representing the same fitted component twice.

Interpretability
----------------

RPTE exposes the complete fitted decision structure at three complementary
levels:

* ``unified_rule_tree()`` returns the nested tree structure, split provenance,
  thresholds, leaf values, support, and downstream coefficients.
* ``unified_rule_table()`` returns a flat review table with one row per active
  leaf rule or direct source term.
* Complexity APIs report compact model units, expanded model inspection units,
  and row-specific instance inspection units.

Tree rows retain their backend identity (sequential or bounded lookahead),
source-feature lineage, split thresholds, support, coefficient, odds
multiplier, and centered contribution. Direct rows identify whether the source
is an original feature, mined pattern, or augmented pair. Canonicalized
leaf-pattern aliases are exposed separately because they are audit records, not
independent fitted coefficients. Governance and LLM
evidence layers consume these same structured records, so dashboard summaries
remain traceable to the fitted model.

Basic use
---------

.. code-block:: python

   from hugiml import HUGIMLClassifier
   from hugiml.rpte_bounded_lookahead_leafwise import (
       LeafWiseBoundedLookaheadRPTEFeatureLR,
   )

   base = HUGIMLClassifier(
       feature_mode="original_plus_patterns",
       adaptive_binning=True,
       L=3,
   )
   X_hug = base.fit_transform(X_train, y_train)

   rpte = LeafWiseBoundedLookaheadRPTEFeatureLR(
       leaf_config="3xD",
       depth=4,
       enable_lookahead="adaptive",
       random_state=42,
   )
   rpte.fit(X_hug, y_train, hugiml_model=base)

   probabilities = rpte.predict_proba(base.transform(X_test))[:, 1]
   rule_table = rpte.unified_rule_table()
   rule_tree = rpte.unified_rule_tree()

Binary and multiclass behavior
------------------------------

The RPTE estimator is a binary classifier. Multiclass use is available through
a one-versus-rest wrapper, giving each class its own inspectable binary RPTE
model. The native extension is required for RPTE tree growth and boosting
operations.

Configuration guidance
----------------------

* Start with ``enable_lookahead="adaptive"`` unless the data structure is
  already known.
* Use ``leaf_config="2xD"`` for the smallest trees and ``3xD`` or ``4xD`` when
  a bounded higher-order search is justified.
* Keep probe-gain and statistical acceptance enabled for ordinary use; they
  limit the chance that a large candidate search admits unstable structures.
* Review both the rule table and instance inspection units before promotion so
  global model size and prediction-level review effort are both visible.
