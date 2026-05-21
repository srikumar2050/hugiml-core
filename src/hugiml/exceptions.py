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

"""Structured exception and warning hierarchy for HUG-IML.

Taxonomy::

    HUGIMLError (base)
    ├── HUGIMLFitError          — any failure during fit()
    │   ├── HUGIMLMiningError   — pattern mining specifically
    │   └── HUGIMLTimeoutError  — max_fit_seconds exceeded
    ├── HUGIMLValidationError   — input data / param validation
    │   ├── HUGIMLSchemaError   — column mismatch at predict time
    │   └── HUGIMLParamError    — bad hyperparameter values / types
    ├── HUGIMLSerializationError — load/save failures
    │   └── HUGIMLVersionError  — schema version incompatibility
    └── HUGIMLPredictionError   — failures during predict/transform

    HUGIMLWarning (base, UserWarning subclass)
    ├── HUGIMLConvergenceWarning — model converged to minimal patterns
    ├── HUGIMLDtypeDriftWarning  — categorical column dtype changed
    ├── HUGIMLRangeWarning       — feature values outside training range
    ├── HUGIMLDegradedWarning    — model degraded due to timeout/memory
    └── HUGIMLDeprecationWarning — deprecated API usage
"""

from __future__ import annotations

__all__ = [
    # Exceptions
    "HUGIMLError",
    "HUGIMLFitError",
    "HUGIMLMiningError",
    "HUGIMLTimeoutError",
    "HUGIMLValidationError",
    "HUGIMLSchemaError",
    "HUGIMLParamError",
    "HUGIMLSerializationError",
    "HUGIMLVersionError",
    "HUGIMLPredictionError",
    # Warnings
    "HUGIMLWarning",
    "HUGIMLConvergenceWarning",
    "HUGIMLDtypeDriftWarning",
    "HUGIMLRangeWarning",
    "HUGIMLDegradedWarning",
    "HUGIMLDeprecationWarning",
]


# =============================================================================
# Exception hierarchy
# =============================================================================


class HUGIMLError(Exception):
    """Base exception for all HUG-IML errors."""


class HUGIMLFitError(HUGIMLError):
    """Raised when fit() fails for any reason."""


class HUGIMLMiningError(HUGIMLFitError):
    """Raised when pattern mining fails or produces zero patterns."""


class HUGIMLTimeoutError(HUGIMLFitError):
    """Raised when fit exceeds max_fit_seconds."""


class HUGIMLValidationError(HUGIMLError, ValueError):
    """Raised when input data or configuration is invalid.

    Inherits from ValueError for backward compatibility with existing
    except-ValueError handlers.
    """


class HUGIMLSchemaError(HUGIMLValidationError):
    """Raised when predict-time data does not match training schema
    (wrong columns, wrong order, wrong count)."""


class HUGIMLParamError(HUGIMLValidationError, TypeError):
    """Raised when hyperparameters have wrong types or values.

    Inherits from both TypeError and ValueError for backward compatibility.
    """


class HUGIMLSerializationError(HUGIMLError):
    """Raised when model save/load fails."""


class HUGIMLVersionError(HUGIMLSerializationError):
    """Raised when loading a model whose schema version is incompatible."""


class HUGIMLPredictionError(HUGIMLError, RuntimeError):
    """Raised when predict/transform fails on a fitted model.

    Inherits from RuntimeError for backward compatibility.
    """


# =============================================================================
# Warning hierarchy
# =============================================================================


class HUGIMLWarning(UserWarning):
    """Base warning for all HUG-IML warnings."""


class HUGIMLConvergenceWarning(HUGIMLWarning):
    """Issued when the model converges to a minimal number of patterns
    (e.g. due to very restrictive G or low-information data)."""


class HUGIMLDtypeDriftWarning(HUGIMLWarning):
    """Issued when a categorical column is passed as numeric at predict time."""


class HUGIMLRangeWarning(HUGIMLWarning):
    """Issued when feature values fall far outside the training range."""


class HUGIMLDegradedWarning(HUGIMLWarning):
    """Issued when the model entered degraded mode due to timeout or
    memory pressure during fit()."""


class HUGIMLDeprecationWarning(HUGIMLWarning, DeprecationWarning):
    """Issued for deprecated API usage."""
