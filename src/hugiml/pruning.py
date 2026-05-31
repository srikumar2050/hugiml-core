# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Regulated "remove / refit / calibrate" workflow for HUG-IML.

EBMs are valued partly because model terms can be inspected and sometimes
edited (e.g. to remove an ethically problematic interaction term).  This
module gives HUG-IML an analogous *controlled editing* workflow that is
rigorous enough for regulated-domain review cycles.

Workflow
--------
1. Inspect patterns via ``clf.feature_importances()`` or ``clf.get_pattern_info()``.
2. Create a ``PatternEditor`` and call ``remove()`` with a list of pattern
   indices (or keyword filters).
3. Call ``refit(X_tr, y_tr)`` to re-train the *downstream* classifier on
   the pruned pattern matrix.  The C++ mining results are unchanged.
4. Optionally call ``calibrate(X_cal, y_cal)`` to wrap the refitted model
   with Platt scaling / isotonic regression.
5. Call ``finalize()`` to get a new classifier instance with the edited
   pattern set baked in, and ``audit_report()`` for a JSON audit trail.

Example
-------
    from hugiml.pruning import PatternEditor

    editor = PatternEditor(clf)
    editor.remove([3, 7, 12], reason="pattern references protected attribute 'gender'")
    editor.remove_by_keyword("income", reason="unstable feature (high PSI)")
    new_clf = editor.refit(X_tr, y_tr).calibrate(X_cal, y_cal).finalize()

    print(editor.audit_report())
    new_clf.predict_proba(X_te)
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

__all__ = [
    "PatternEditor",
    "RemovalRecord",
]


# ---------------------------------------------------------------------------
# Audit entry
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RemovalRecord:
    """Audit record for a single pattern-removal action."""

    timestamp: str
    pattern_indices: list[int]
    pattern_labels: list[str]
    reason: str
    removed_by: str


# ---------------------------------------------------------------------------
# PatternEditor
# ---------------------------------------------------------------------------


class PatternEditor:
    """Controlled pattern editing with full audit trail.

    Parameters
    ----------
    clf : fitted HUGIMLClassifierNative
        The *original* model.  This object is **not** mutated; all edits
        produce a fresh copy stored internally.
    operator_name : str
        Human-readable identifier of the person/process making the edits
        (for the audit trail).
    """

    def __init__(self, clf: Any, operator_name: str = "analyst") -> None:
        if not hasattr(clf, "patterns_"):
            raise RuntimeError("Classifier must be fitted before creating a PatternEditor.")
        self._orig_clf = clf
        self._operator = operator_name
        self._working_clf: Any = copy.deepcopy(clf)
        self._audit_log: list[RemovalRecord] = []
        self._calibrated: bool = False
        self._calibration_method: str | None = None
        self._finalized: bool = False
        self._refitted: bool = False

    # ------------------------------------------------------------------
    # Removal operations
    # ------------------------------------------------------------------

    def remove(
        self,
        pattern_indices: list[int],
        reason: str = "unspecified",
    ) -> PatternEditor:
        """Remove patterns by index (0-based, relative to the *current* working set).

        Parameters
        ----------
        pattern_indices : list of int
            Indices into the *current* pattern list.  Use
            ``list_patterns()`` to preview indices.
        reason : str
            Audit reason (e.g. 'protected attribute', 'operationally invalid').

        Returns
        -------
        self  (for method chaining)
        """
        self._check_not_finalized()
        clf = self._working_clf
        n = len(clf.patterns_)
        valid = [i for i in pattern_indices if 0 <= i < n]
        invalid = set(pattern_indices) - set(valid)
        if invalid:
            warnings.warn(f"PatternEditor.remove: ignoring out-of-range indices {invalid}.")

        if not valid:
            return self

        all_labels = clf.get_hug_features()
        removed_labels = [all_labels[i] for i in valid]
        keep_mask = np.ones(n, dtype=bool)
        for i in valid:
            keep_mask[i] = False
        keep_idx = np.where(keep_mask)[0]

        # Update patterns and pattern matrix
        clf.patterns_ = [clf.patterns_[i] for i in keep_idx]
        if hasattr(clf, "x_train_hup_") and clf.x_train_hup_ is not None:
            clf.x_train_hup_ = clf.x_train_hup_[:, keep_idx]

        self._audit_log.append(
            RemovalRecord(
                timestamp=datetime.datetime.utcnow().isoformat(),
                pattern_indices=list(valid),
                pattern_labels=removed_labels,
                reason=reason,
                removed_by=self._operator,
            )
        )
        return self

    def remove_by_keyword(
        self,
        keyword: str,
        reason: str = "keyword match",
        case_sensitive: bool = False,
    ) -> PatternEditor:
        """Remove all patterns whose label contains ``keyword``.

        Parameters
        ----------
        keyword : str
        reason : str
        case_sensitive : bool

        Returns
        -------
        self
        """
        self._check_not_finalized()
        labels = self._working_clf.get_hug_features()
        if not case_sensitive:
            matches = [i for i, lbl in enumerate(labels) if keyword.lower() in lbl.lower()]
        else:
            matches = [i for i, lbl in enumerate(labels) if keyword in lbl]
        if not matches:
            warnings.warn(f"PatternEditor.remove_by_keyword: no patterns matched '{keyword}'.")
            return self
        return self.remove(matches, reason=f"{reason} (keyword='{keyword}')")

    def remove_low_support(
        self,
        min_support: float = 0.01,
        reason: str = "support below threshold",
    ) -> PatternEditor:
        """Remove patterns with training support below ``min_support``.

        Parameters
        ----------
        min_support : float
            Minimum fraction of training samples (0 to 1).
        reason : str

        Returns
        -------
        self
        """
        self._check_not_finalized()
        clf = self._working_clf
        if not hasattr(clf, "x_train_hup_"):
            raise RuntimeError("x_train_hup_ not available — cannot filter by support.")
        n_train = clf.x_train_hup_.shape[0]
        supports = np.asarray(clf.x_train_hup_.sum(axis=0)).ravel() / n_train
        to_remove = [i for i, s in enumerate(supports) if s < min_support]
        return self.remove(to_remove, reason=f"{reason} (min_support={min_support})")

    # ------------------------------------------------------------------
    # Refit downstream classifier
    # ------------------------------------------------------------------

    def refit(
        self,
        X_tr: Any,
        y_tr: Any,
        estimator: Any = None,
    ) -> PatternEditor:
        """Refit the downstream classifier on the (pruned) pattern matrix.

        The HUG mining results (``patterns_``) are unchanged; only the downstream
        ``Pipeline`` (``model_``) is replaced.

        Parameters
        ----------
        X_tr : array-like or DataFrame
            Training data (should be the same split used to fit the original model).
        y_tr : array-like
        estimator : sklearn estimator, optional
            If None, uses the original downstream estimator class with the same
            hyperparameters.

        Returns
        -------
        self
        """
        self._check_not_finalized()
        clf = self._working_clf
        import copy as _copy

        from sklearn.pipeline import Pipeline

        # Build pattern matrix for training data.  After remove(), the working
        # classifier already carries the original training HUP matrix with the
        # same columns pruned.  Reuse it when the caller passes the same-row
        # training split; rebuilding it by scanning every pattern can dominate
        # refit time for large adaptive models.
        y_arr = np.asarray(y_tr, dtype=np.int64)
        cached_hup = getattr(clf, "x_train_hup_", None)
        if cached_hup is not None and cached_hup.shape[0] == len(y_arr):
            clf.x_train_hup_ = csr_matrix(cached_hup, dtype=np.float32)
        else:
            hup_tr = clf.transform(X_tr)
            clf.x_train_hup_ = csr_matrix(hup_tr, dtype=np.float32)

        if estimator is not None:
            new_est = _copy.deepcopy(estimator)
        else:
            orig_clf_step = self._orig_clf.model_.named_steps.get("clf")
            new_est = _copy.deepcopy(orig_clf_step)

        new_model = Pipeline([("clf", new_est)])
        new_model.fit(clf.x_train_hup_, y_arr)
        clf.model_ = new_model
        self._refitted = True

        self._calibrated = False
        self._calibration_method = None
        return self

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        X_cal: Any,
        y_cal: Any,
        method: str = "isotonic",
    ) -> PatternEditor:
        """Wrap the refitted downstream model with probability calibration.

        Uses ``sklearn.calibration.CalibratedClassifierCV`` applied *post-fit*
        to a calibration set that should be held out from both training and test.

        Parameters
        ----------
        X_cal : array-like or DataFrame
        y_cal : array-like
        method : {'sigmoid', 'isotonic'}

        Returns
        -------
        self
        """
        self._check_not_finalized()
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline

        clf = self._working_clf
        hup_cal = csr_matrix(clf.transform(X_cal), dtype=np.float32)
        y_arr = np.asarray(y_cal, dtype=np.int64)

        inner_clf = clf.model_.named_steps["clf"]
        # sklearn >= 1.8 removed cv="prefit"; cv=None with ensemble=False
        # is the documented replacement, preserving pre-fitted behaviour.
        try:
            cal_clf = CalibratedClassifierCV(inner_clf, cv="prefit", method=method)
            cal_clf.fit(hup_cal, y_arr)
        except (ValueError, TypeError):
            cal_clf = CalibratedClassifierCV(inner_clf, cv=None, ensemble=False, method=method)
            cal_clf.fit(hup_cal, y_arr)

        clf.model_ = Pipeline([("clf", cal_clf)])
        self._calibrated = True
        self._calibration_method = method
        return self

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> Any:
        """Return the edited classifier as a new standalone instance.

        After calling ``finalize()``, further edits on this editor are
        blocked.  The returned object is a fully independent copy.

        Returns
        -------
        HUGIMLClassifierNative (edited copy)
        """
        if not self._refitted:
            raise RuntimeError(
                "Call refit() before finalize() — the downstream model must be retrained "
                "after pattern removal."
            )
        self._finalized = True
        return copy.deepcopy(self._working_clf)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def list_patterns(self) -> pd.DataFrame:
        """Return a DataFrame of the *current* (possibly pruned) patterns.

        Columns: idx, pattern, coefficient (if available), support.
        """
        clf = self._working_clf
        labels = clf.get_hug_features()
        n_train = clf.x_train_hup_.shape[0] if hasattr(clf, "x_train_hup_") else None

        try:
            imp = clf.feature_importances()
            coef_map = dict(zip(imp["pattern"], imp["coefficient"]))
            sup_map = dict(zip(imp["pattern"], imp["support"]))
        except Exception:
            coef_map = {}
            sup_map = {}

        rows = []
        for i, lbl in enumerate(labels):
            sup = sup_map.get(lbl, None)
            if sup is None and n_train and hasattr(clf, "x_train_hup_"):
                sup = float(clf.x_train_hup_[:, i].sum()) / n_train
            rows.append(
                {
                    "idx": i,
                    "pattern": lbl,
                    "coefficient": coef_map.get(lbl, float("nan")),
                    "support": round(sup, 4) if sup is not None else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    def diff(self) -> dict:
        """Return a summary of changes made relative to the original model.

        Returns
        -------
        dict with keys: n_original, n_current, n_removed, removed_patterns
        """
        original_labels = set(self._orig_clf.get_hug_features())
        current_labels = set(self._working_clf.get_hug_features())
        removed = sorted(original_labels - current_labels)
        return {
            "n_original": len(original_labels),
            "n_current": len(current_labels),
            "n_removed": len(removed),
            "removed_patterns": removed,
        }

    # ------------------------------------------------------------------
    # Audit report
    # ------------------------------------------------------------------

    def audit_report(self, indent: int = 2) -> str:
        """Return a JSON string describing all edits made.

        The report includes operator name, timestamps, reasons, and the
        diff summary.
        """
        report = {
            "operator": self._operator,
            "generated_at": datetime.datetime.utcnow().isoformat(),
            "diff": self.diff(),
            "calibration": {
                "applied": self._calibrated,
                "method": self._calibration_method,
            },
            "removals": [dataclasses.asdict(r) for r in self._audit_log],
        }
        return json.dumps(report, indent=indent, default=str)

    def save_audit_report(self, path: str) -> None:
        """Write the audit report to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.audit_report())

    # ------------------------------------------------------------------
    # Context manager support (optional)
    # ------------------------------------------------------------------

    def __enter__(self) -> PatternEditor:
        return self

    def __exit__(self, *exc) -> None:
        pass  # Caller is responsible for calling finalize()

    def __repr__(self) -> str:
        diff = self.diff()
        return (
            f"PatternEditor("
            f"original={diff['n_original']}, "
            f"current={diff['n_current']}, "
            f"removed={diff['n_removed']}, "
            f"calibrated={self._calibrated})"
        )

    # ------------------------------------------------------------------

    def _check_not_finalized(self) -> None:
        if self._finalized:
            raise RuntimeError(
                "PatternEditor has already been finalized.  "
                "Create a new PatternEditor to make further edits."
            )
