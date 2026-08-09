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

"""Internal sklearn compatibility shims.

Centralises version-dependent sklearn API differences so the rest of the
package can import from here without per-module try/except blocks.

Compatibility contract
----------------------
HUGIMLClassifierNative targets the following sklearn surface:

* ``check_array`` / ``check_X_y`` — present in sklearn >= 1.0; replaced by
  ``validate_data`` in sklearn >= 1.6 (1.6 retains the old names as shims
  but marks them deprecated; they are expected to be removed in sklearn 2.0).
  This module provides a stable ``check_array`` / ``check_X_y`` API for any
  supported sklearn version.

* ``check_is_fitted`` — stable across all supported versions.

* ``BaseEstimator`` / ``ClassifierMixin`` / ``TransformerMixin`` — stable.

* ``__sklearn_tags__`` — introduced in sklearn 1.6 to replace ``_get_tags``.
  ``HUGIMLClassifierNative.__sklearn_tags__`` handles both APIs.

* ``LogisticRegression(penalty=...)`` vs ``l1_ratio=...`` — sklearn >= 1.8
  deprecates the ``penalty`` string in favor of expressing the L1/L2 mixture
  through ``l1_ratio`` for every solver (removal targeted for sklearn 1.10),
  and treats ``l1_ratio`` accordingly once ``penalty`` is left unset. On
  sklearn < 1.8, ``l1_ratio`` is honored only when ``penalty="elasticnet"``;
  passing it with any other (or default) ``penalty`` is silently ignored
  with a ``UserWarning`` and the fit falls back to ``penalty``'s own value
  -- so the two spellings are not interchangeable and must be chosen by
  installed version, not simply swapped. ``logistic_penalty_kwargs``
  provides the correct one for the installed version.

Supported sklearn range: >= 1.0  (tested through current stable).

Unsupported operations are logged at DEBUG level and raise ImportError as
usual so callers can decide how to handle them.
"""

from __future__ import annotations

import logging
from typing import Any

from packaging.version import Version as _V

__all__ = [
    "check_X_y",
    "check_array",
    "sklearn_version",
    "SKLEARN_VERSION",
    "logistic_penalty_kwargs",
    "liblinear_penalty_kwargs",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Introspect installed sklearn version once at import time
# ---------------------------------------------------------------------------

try:
    import sklearn as _sklearn

    sklearn_version: str = _sklearn.__version__
    SKLEARN_VERSION: _V = _V(sklearn_version)
except Exception:  # pragma: no cover
    sklearn_version = "0.0.0"
    SKLEARN_VERSION = _V("0.0.0")
    logger.debug("Could not determine sklearn version.", exc_info=True)

_SKLEARN_HAS_VALIDATE_DATA: bool = SKLEARN_VERSION >= _V("1.6")

logger.debug(
    "sklearn version %s detected; validate_data shim active=%s",
    sklearn_version,
    _SKLEARN_HAS_VALIDATE_DATA,
)

# ---------------------------------------------------------------------------
# check_array / check_X_y
# ---------------------------------------------------------------------------

if not _SKLEARN_HAS_VALIDATE_DATA:
    # sklearn 1.0 – 1.5: the classic helpers are present and not deprecated.
    try:
        from sklearn.utils.validation import check_array, check_X_y

        logger.debug("Using sklearn.utils.validation.check_array / check_X_y directly.")
    except ImportError as _exc:  # pragma: no cover
        raise ImportError(
            "sklearn.utils.validation.check_array / check_X_y not found.  "
            "Ensure scikit-learn >= 1.0 is installed."
        ) from _exc
else:
    # sklearn >= 1.6: check_array / check_X_y are deprecated shims around
    # validate_data.  We re-implement the shim ourselves to avoid deprecation
    # warnings in user code without depending on internal shim stability.
    try:
        from sklearn.utils.validation import check_array, check_X_y

        logger.debug(
            "Using sklearn %s check_array / check_X_y (deprecated shims; "
            "will migrate to validate_data when shims are removed).",
            sklearn_version,
        )
    except ImportError:
        # The shims have been removed — implement via validate_data.
        logger.debug(
            "sklearn %s: check_array / check_X_y removed; implementing via validate_data.",
            sklearn_version,
        )
        from sklearn.base import BaseEstimator as _BE
        from sklearn.utils.validation import validate_data as _vd

        class _Stub(_BE):
            pass

        _stub = _Stub()

        def check_X_y(X: Any, y: Any, **kw: Any) -> Any:
            return _vd(_stub, X, y, reset=False, **kw)

        def check_array(X: Any, **kw: Any) -> Any:
            return _vd(_stub, X, reset=False, **kw)


# ---------------------------------------------------------------------------
# LogisticRegression penalty / l1_ratio (see module docstring)
# ---------------------------------------------------------------------------

_SKLEARN_DEPRECATES_PENALTY_STRING: bool = SKLEARN_VERSION >= _V("1.8")

# solver="liblinear" only ever supports penalty in {"l1", "l2"}.
_LIBLINEAR_PENALTY_TO_L1_RATIO = {"l1": 1.0, "l2": 0.0}


def logistic_penalty_kwargs(penalty: str) -> dict[str, Any]:
    """Return version-appropriate LogisticRegression regularization keywords.

    ``penalty`` must be ``"l1"`` or ``"l2"``; any other value is passed
    through as ``{"penalty": penalty}`` unchanged and left to raise
    sklearn's own error, exactly as it would without this shim.
    """
    if not _SKLEARN_DEPRECATES_PENALTY_STRING:
        return {"penalty": penalty}
    l1_ratio = _LIBLINEAR_PENALTY_TO_L1_RATIO.get(penalty)
    if l1_ratio is None:
        return {"penalty": penalty}
    return {"l1_ratio": l1_ratio}


def liblinear_penalty_kwargs(penalty: str) -> dict[str, Any]:
    """Compatibility alias for callers using the liblinear solver."""
    return logistic_penalty_kwargs(penalty)
