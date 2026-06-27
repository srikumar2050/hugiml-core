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

"""Versioned serialization and SBOM generation for HUGIMLClassifierNative.

Format (v3 — default)
---------------------
A ZIP archive containing JSON manifests and NumPy array bundles.  No
``pickle`` is required to round-trip the model, eliminating the gadget-chain
attack surface that exists in any pickle-based format.

Archive layout::

    manifest.json          – format_version, schema_version, timestamp
    clf_init.json          – __init__ hyperparameters
    clf_fit.json           – scalar / list fitted attributes
    patterns.json          – list of {utility, items, ig} dicts
    arrays.npz             – cat_cols_mask_, is_int_mask_, classes_
    td_config.json         – TransactionDataWrapper non-array state
    td_arrays.npz          – TransactionDataWrapper numpy arrays
    estimator.json         – downstream estimator class + parameters
    estimator_arrays.npz   – downstream estimator numpy arrays
    hmac.sig               – HMAC-SHA256 over all content files (hex)

Authentication
--------------
Set ``HUGIML_MODEL_HMAC_KEY`` (hex-encoded, 32+ bytes) before saving or
loading.  Files saved without a key have an all-zero ``hmac.sig`` and can
still be loaded unless ``HUGIML_REQUIRE_MODEL_HMAC=true`` is set.

Backward compatibility (v1/v2)
-------------------------------
Models saved with schema version 1 or 2 (the legacy HMAC-pickle format) are
still loadable via a restricted Unpickler that permits only known HUG-IML and
sklearn modules.  v1/v2 writing is not supported.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import os
import struct
import time
import zipfile
from typing import Any

import numpy as np
import pandas as pd

from hugiml.exceptions import HUGIMLSerializationError, HUGIMLVersionError

logger = logging.getLogger(__name__)

__all__ = [
    "MODEL_SCHEMA_VERSION",
    "MIN_SCHEMA_VERSION",
    "save_model",
    "load_model",
    "generate_sbom",
]

# Schema v5 adds execution_mode and lightweight retained training-matrix
# shape/nnz metadata to clf_fit.json.  Loading remains backward-compatible
# with v1-v4 because new fields are restored with .get(..., None).
MODEL_SCHEMA_VERSION: int = 6
MIN_SCHEMA_VERSION: int = 1

# ── Legacy (v1/v2) pickle-envelope constants ──────────────────────────────────
_LEGACY_MAGIC = b"HUGI"
_HMAC_LEN = 32
_LEGACY_HEADER_LEN = 4 + 4 + _HMAC_LEN

# ── Modules / classes permitted by the legacy restricted Unpickler ────────────
_SAFE_MODULES = (
    "builtins",
    "collections",
    "numpy",
    "numpy.core",
    "numpy._core",
    "scipy.sparse",
    "sklearn",
    "pandas",
    "hugiml",
    "hugiml.classifier",
    "hugiml.monitoring",
    "hugiml.exceptions",
    "hugiml.serialization",
)
_SAFE_TYPES = frozenset(
    [
        "HUGIMLClassifierNative",
        "HUGIMLClassifier",
        "_TransactionDataWrapper",
        "FitMetadata",
        "PredictionMonitor",
        "DriftDetector",
        "NativeAugmentedPairTransformBlock",
    ]
)


# =============================================================================
# Environment helpers
# =============================================================================


def _require_hmac() -> bool:
    return os.environ.get("HUGIML_REQUIRE_MODEL_HMAC", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_hmac_key() -> bytes | None:
    raw = os.environ.get("HUGIML_MODEL_HMAC_KEY", "")
    if not raw:
        return None
    try:
        key = bytes.fromhex(raw)
    except ValueError as exc:
        raise HUGIMLSerializationError(
            "HUGIML_MODEL_HMAC_KEY must be a hex-encoded byte string."
        ) from exc
    if len(key) < 16:
        raise HUGIMLSerializationError(
            "HUGIML_MODEL_HMAC_KEY must be at least 16 bytes (32 hex chars)."
        )
    return key


# =============================================================================
# v3 – ZIP / JSON / NumPy format (no pickle)
# =============================================================================


def _json_dumps(obj: Any) -> bytes:
    """Serialize *obj* to UTF-8 JSON bytes, converting numpy scalars."""

    def _default(o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(obj, default=_default, separators=(",", ":")).encode()


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    """Return the binary content of a numpy .npz file without touching disk."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)  # type: ignore[arg-type]
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Downstream estimator (de)serialization
# ---------------------------------------------------------------------------


def _serialize_logreg(est: Any) -> tuple[dict, dict[str, np.ndarray]]:
    """Serialize a fitted LogisticRegression without pickle."""
    config: dict[str, Any] = {
        "class": "sklearn.linear_model.LogisticRegression",
        "init_params": {
            k: v
            for k, v in est.get_params().items()
            if not callable(v) and v is not None or k in {"class_weight", "random_state"}
        },
        "n_features_in_": int(est.n_features_in_),
    }
    # n_iter_ may be absent if the estimator was never actually fitted
    if hasattr(est, "n_iter_"):
        config["n_iter_list"] = [int(x) for x in est.n_iter_]
    arrays: dict[str, np.ndarray] = {
        "coef_": np.asarray(est.coef_, dtype=np.float64),
        "intercept_": np.asarray(est.intercept_, dtype=np.float64),
        "classes_": np.asarray(est.classes_),
    }
    return config, arrays


def _deserialize_logreg(config: dict, arrays: dict[str, np.ndarray]) -> Any:
    # Filter init_params to only those the current sklearn version accepts
    import inspect

    from sklearn.linear_model import LogisticRegression

    valid = set(inspect.signature(LogisticRegression.__init__).parameters)
    init_params = {k: v for k, v in config.get("init_params", {}).items() if k in valid}
    est = LogisticRegression(**init_params)
    est.coef_ = arrays["coef_"]
    est.intercept_ = arrays["intercept_"]
    est.classes_ = arrays["classes_"]
    est.n_features_in_ = config["n_features_in_"]
    if "n_iter_list" in config:
        est.n_iter_ = np.array(config["n_iter_list"], dtype=np.int32)
    return est


def _serialize_pipeline(est: Any) -> tuple[dict, dict[str, np.ndarray]]:
    """Serialize a sklearn Pipeline whose final step is LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if not isinstance(est, Pipeline):
        raise HUGIMLSerializationError(f"_serialize_pipeline called on {type(est).__name__}")

    steps_config = []
    arrays_all: dict[str, np.ndarray] = {}
    for name, step in est.steps:
        if isinstance(step, LogisticRegression):
            sc, sa = _serialize_logreg(step)
            steps_config.append({"name": name, "estimator": sc})
            for k, v in sa.items():
                arrays_all[f"{name}__{k}"] = v
        else:
            logger.debug(
                "Pipeline step '%s' (%s) cannot be natively serialized; "
                "falling back to restricted pickle for this step.",
                name,
                type(step).__name__,
            )
            import pickle  # nosec B403 – payload signed by HMAC on save

            payload = pickle.dumps(step, protocol=5)
            sc = {
                "class": type(step).__module__ + "." + type(step).__qualname__,
                "_pickle_fallback": True,
            }
            steps_config.append({"name": name, "estimator": sc})
            arrays_all[f"{name}__pickle_payload"] = np.frombuffer(payload, dtype=np.uint8)

    config = {"class": "sklearn.pipeline.Pipeline", "steps": steps_config}
    return config, arrays_all


def _deserialize_pipeline(config: dict, arrays: dict[str, np.ndarray]) -> Any:
    from sklearn.pipeline import Pipeline

    steps = []
    for step_cfg in config["steps"]:
        name = step_cfg["name"]
        est_cfg = step_cfg["estimator"]
        if est_cfg.get("_pickle_fallback"):
            payload = arrays[f"{name}__pickle_payload"].tobytes()
            step_est = _safe_unpickle(payload)
        elif est_cfg.get("class") == "sklearn.linear_model.LogisticRegression":
            step_arrays = {
                k.removeprefix(f"{name}__"): v
                for k, v in arrays.items()
                if k.startswith(f"{name}__")
            }
            step_est = _deserialize_logreg(est_cfg, step_arrays)
        else:
            raise HUGIMLSerializationError(
                f"Cannot deserialize pipeline step '{name}' of class {est_cfg.get('class')}."
            )
        steps.append((name, step_est))
    return Pipeline(steps)


def _serialize_estimator(
    est: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Serialize a fitted downstream estimator to (config_dict, arrays_dict).

    Supports LogisticRegression natively.  Pipeline with LogisticRegression
    final step is also supported.  All other estimators fall back to a
    restricted pickle stored as a uint8 byte array (logged at DEBUG level).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if isinstance(est, LogisticRegression):
        return _serialize_logreg(est)
    if isinstance(est, Pipeline):
        return _serialize_pipeline(est)

    # Generic fallback: restricted pickle
    logger.debug(
        "Downstream estimator %s is not natively serializable; "
        "using restricted-pickle fallback.  Consider using LogisticRegression "
        "for a fully pickle-free model artifact.",
        type(est).__name__,
    )
    import pickle  # nosec B403 – controlled: payload is HMAC-signed on save

    payload = pickle.dumps(est, protocol=5)
    config: dict[str, Any] = {
        "class": type(est).__module__ + "." + type(est).__qualname__,
        "_pickle_fallback": True,
        "n_features_in_": getattr(est, "n_features_in_", None),
    }
    arrays: dict[str, np.ndarray] = {"pickle_payload": np.frombuffer(payload, dtype=np.uint8)}
    return config, arrays


def _deserialize_estimator(config: dict, arrays: dict[str, np.ndarray]) -> Any:
    if config.get("_pickle_fallback"):
        payload = arrays["pickle_payload"].tobytes()
        return _safe_unpickle(payload)
    cls_name = config.get("class", "")
    if cls_name == "sklearn.linear_model.LogisticRegression":
        return _deserialize_logreg(config, arrays)
    if cls_name == "sklearn.pipeline.Pipeline":
        return _deserialize_pipeline(config, arrays)
    raise HUGIMLSerializationError(f"Cannot deserialize estimator of class '{cls_name}'.")


# ---------------------------------------------------------------------------
# TransactionDataWrapper (de)serialization
# ---------------------------------------------------------------------------


def _serialize_td(td: Any) -> tuple[dict, dict[str, np.ndarray]]:
    """Serialize a _TransactionDataWrapper to (config, arrays)."""
    state: dict = td.__getstate__()

    config: dict[str, Any] = {
        "_cpp_bn2id": {str(k): int(v) for k, v in state.get("_cpp_bn2id", {}).items()},
        "_cpp_bkey_stride": int(state.get("_cpp_bkey_stride", 1)),
        "_cpp_nb_col": [int(x) for x in state.get("_cpp_nb_col", [])],
        "_cpp_is_cat": [bool(x) for x in state.get("_cpp_is_cat", [])],
        "_cpp_is_int": [bool(x) for x in state.get("_cpp_is_int", [])],
        "_cpp_cat_categories": [
            [(str(v) if not isinstance(v, (int, float, bool)) else v) for v in cat]
            for cat in state.get("_cpp_cat_categories", [])
        ],
        "item_map": {str(k): str(v) for k, v in state.get("item_map", {}).items()},
        "item_twu": [float(x) for x in state.get("item_twu", [])],
        "nb_col": [int(x) for x in state.get("nb_col", [])],
    }

    arrays: dict[str, np.ndarray] = {}
    if "_cpp_col_min" in state and state["_cpp_col_min"] is not None:
        arrays["col_min"] = np.asarray(state["_cpp_col_min"], dtype=np.float64)
    if "_cpp_col_range" in state and state["_cpp_col_range"] is not None:
        arrays["col_range"] = np.asarray(state["_cpp_col_range"], dtype=np.float64)
    all_edges = state.get("_cpp_all_edges", [])
    config["n_edges"] = len(all_edges)
    for i, edge in enumerate(all_edges):
        arrays[f"edge_{i}"] = np.asarray(edge, dtype=np.float64)

    return config, arrays


def _deserialize_td(config: dict, arrays: dict[str, np.ndarray]) -> Any:
    """Reconstruct a _TransactionDataWrapper state dict and restore via __setstate__."""
    from hugiml.classifier import _TransactionDataWrapper

    state: dict[str, Any] = {
        "_cpp_bn2id": {int(k): int(v) for k, v in config["_cpp_bn2id"].items()},
        "_cpp_bkey_stride": config["_cpp_bkey_stride"],
        "_cpp_nb_col": config["_cpp_nb_col"],
        "_cpp_is_cat": config["_cpp_is_cat"],
        "_cpp_is_int": config["_cpp_is_int"],
        "_cpp_cat_categories": config["_cpp_cat_categories"],
        "item_map": {int(k): str(v) for k, v in config["item_map"].items()},
        "item_twu": config["item_twu"],
        "nb_col": config["nb_col"],
        "_cpp_col_min": arrays.get("col_min", np.array([], dtype=np.float64)),
        "_cpp_col_range": arrays.get("col_range", np.array([], dtype=np.float64)),
        "_cpp_all_edges": [arrays[f"edge_{i}"] for i in range(config.get("n_edges", 0))],
    }
    # Populate compat aliases expected by the Python fallback predict path
    state["col_min"] = state["_cpp_col_min"]
    state["col_range"] = state["_cpp_col_range"]
    state["all_edges"] = state["_cpp_all_edges"]

    td = object.__new__(_TransactionDataWrapper)
    td.__setstate__(state)
    return td


# ---------------------------------------------------------------------------
# HMAC helpers
# ---------------------------------------------------------------------------


def _compute_archive_hmac(key: bytes, member_contents: dict[str, bytes]) -> str:
    """HMAC-SHA256 over all archive members (excluding hmac.sig).

    The digest is computed over a canonical string:
    ``"<name>:<sha256hex>\\n"`` for each member, sorted by name.
    This is deterministic, position-independent, and tamper-evident.
    """
    entries = sorted(
        f"{name}:{hashlib.sha256(content).hexdigest()}"
        for name, content in member_contents.items()
        if name != "hmac.sig"
    )
    message = "\n".join(entries).encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _verify_archive_hmac(
    key: bytes,
    member_contents: dict[str, bytes],
    stored_sig: str,
) -> bool:
    expected = _compute_archive_hmac(key, member_contents)
    return hmac.compare_digest(stored_sig, expected)


# =============================================================================
# Public API
# =============================================================================


def save_model(clf: Any, path: str | os.PathLike) -> None:
    """Persist a fitted classifier to a v3 ZIP/JSON/NumPy model file.

    Parameters
    ----------
    clf : HUGIMLClassifierNative
        A fitted classifier.
    path : str or Path

    Raises
    ------
    HUGIMLSerializationError
        When the model is unfitted, a component cannot be serialized, or the
        write fails.
    """
    if not hasattr(clf, "patterns_"):
        raise HUGIMLSerializationError("Cannot save an unfitted model.  Call fit() first.")

    # ── 1. Collect all archive members as bytes ───────────────────────────────
    members: dict[str, bytes] = {}

    manifest = {
        "format_version": MODEL_SCHEMA_VERSION,
        "schema_version": MODEL_SCHEMA_VERSION,
        "algorithm": "HMAC-SHA256",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hugiml_version": _hugiml_version(),
    }
    members["manifest.json"] = _json_dumps(manifest)

    # Init params
    members["clf_init.json"] = _json_dumps(clf.get_params())

    # Fitted scalar / list state
    fit_state: dict[str, Any] = {
        "n_features_in_": int(clf.n_features_in_),
        "classes_list": clf.classes_.tolist(),
        "feature_names_in_": list(clf.feature_names_in_) if clf.feature_names_in_ else None,
        "binary_categorical_cols": list(getattr(clf, "binary_categorical_cols_", []) or []),
        "_native_available_": bool(getattr(clf, "_native_available_", True)),
        "_degraded_reason": getattr(clf, "_degraded_reason", None),
        "n_categories_": getattr(clf, "n_categories_", None),
        "feature_mode": getattr(clf, "feature_mode", "patterns_only"),
        "execution_mode": getattr(clf, "execution_mode", "audit"),
        "original_numeric_cols": getattr(clf, "_original_numeric_cols_", []),
        "original_cat_cols": getattr(clf, "_original_cat_cols_", []),
        "original_dummy_columns": getattr(clf, "_original_dummy_columns_", []),
        "original_feature_names_downstream": getattr(
            clf, "_original_feature_names_downstream_", []
        ),
        "topk_budget_strict": bool(getattr(clf, "topk_budget_strict", False)),
        "downstream_feature_names_full": getattr(clf, "_downstream_feature_names_full_", []),
        "strict_topk_selected_feature_names": getattr(
            clf, "_strict_topk_selected_feature_names_", []
        ),
        "original_selected_feature_names_downstream": getattr(
            clf, "_original_selected_feature_names_downstream_", None
        ),
        "original_feature_names_downstream_full": getattr(
            clf, "_original_feature_names_downstream_full_", None
        ),
        "strict_topk_applied_during_construction": bool(
            getattr(clf, "_strict_topk_applied_during_construction_", False)
        ),
        "training_pattern_matrix_shape": getattr(clf, "_training_pattern_matrix_shape_", None),
        "training_pattern_matrix_nnz": getattr(clf, "_training_pattern_matrix_nnz_", None),
        "training_downstream_matrix_shape": getattr(
            clf, "_training_downstream_matrix_shape_", None
        ),
        "training_downstream_matrix_nnz": getattr(clf, "_training_downstream_matrix_nnz_", None),
        "fallback_state": {
            "active": bool(getattr(clf, "fallback_active_", False)),
            "strategy": getattr(clf, "fallback_strategy_", None),
            "reason": getattr(clf, "fallback_reason_", None),
            "majority_class": getattr(clf, "fallback_majority_class_", None),
            "n_samples": getattr(clf, "fallback_n_samples_", None),
        },
    }
    if hasattr(clf, "fit_metadata_") and clf.fit_metadata_ is not None:
        import dataclasses

        fit_state["fit_metadata_"] = dataclasses.asdict(clf.fit_metadata_)
    # ── Missing value handling state ────────────────────────────────────
    # _missing_col_edges_ stores quantile edges for columns that had NaN/Inf
    # in training data.  Serialised as float lists for JSON compatibility.
    # Schema v5 writes this field unconditionally.  An empty dict explicitly
    # records the clean-training-data contract: no numeric column required
    # Python-side missing-value pre-binning, so loaded models should let new
    # test-time NaN/Inf in those columns be skipped by the native numeric path.
    missing_edges_state = getattr(clf, "_missing_col_edges_", {}) or {}
    fit_state["missing_col_edges"] = {
        name: edges.tolist() for name, edges in missing_edges_state.items()
    }
    # ─────────────────────────────────────────────────────────────────────
    # ── Adaptive binning state ──────────────────────────────────────────
    # _bin_edges_ (dict[str, np.ndarray]) is serialised as JSON-compatible
    # lists.  per_feature_b_ and ig_scores_ are plain dicts.  JSON object
    # keys must be strings so ig_scores_ int keys are stringified here and
    # converted back to int in load_model.
    if getattr(clf, "adaptive_binning", False) and getattr(clf, "_bin_edges_", None):
        fit_state["adaptive_binning_state"] = {
            "bin_edges": {name: edges.tolist() for name, edges in clf._bin_edges_.items()},
            "per_feature_b": dict(clf.per_feature_b_),
            "ig_scores": {
                name: {str(b): float(v) for b, v in sc.items()}
                for name, sc in clf.ig_scores_.items()
            },
        }
    # ────────────────────────────────────────────────────────────────────
    # ── augmented pair transform state ──────────────────────────
    # Persist the public augmentation parameters and fitted transform catalog
    # so versioned save/load can round-trip predictions without relying on
    # pickle.  The augmented pair block is reconstructed during load using
    # these JSON fields plus scaler arrays stored in arrays.npz.
    aug_block = getattr(clf, "_augmented_pair_block_", None)
    fit_state["augmented_pair_state"] = {
        "augmented_pair_transforms": bool(getattr(clf, "augmented_pair_transforms", True)),
        "augmented_pair_mode": str(getattr(clf, "augmented_pair_mode", "interaction_information")),
        "aug_feature_size": int(getattr(clf, "aug_feature_size", 10)),
        "ii_partner_size": getattr(clf, "ii_partner_size", None),
        "max_pair_features": int(getattr(clf, "max_pair_features", 10)),
        "enabled": bool(getattr(clf, "augmented_pair_transforms_enabled_", False)),
        "config": getattr(clf, "augmented_pair_config_", {"enabled": False}),
        "selected_features": getattr(clf, "augmented_pair_selected_features_", []),
        "transforms": getattr(clf, "augmented_pair_transforms_", []),
        "block_state": None,
    }
    if aug_block is not None:
        fit_state["augmented_pair_state"]["block_state"] = {
            "augmented_pair_mode": str(
                getattr(aug_block, "augmented_pair_mode", "interaction_information")
            ),
            "aug_feature_size": int(
                getattr(aug_block, "aug_feature_size", getattr(clf, "aug_feature_size", 10))
            ),
            "ii_partner_size": getattr(aug_block, "ii_partner_size", None),
            "max_pair_features": int(
                getattr(aug_block, "max_pair_features", getattr(clf, "max_pair_features", 10))
            ),
            "budget_topK": getattr(aug_block, "budget_topK", None),
            "selected_aug_features": list(getattr(aug_block, "selected_aug_features_", [])),
            "selected_aug_scores": dict(getattr(aug_block, "selected_aug_scores_", {})),
            "augmented_pair_source_scores": list(
                getattr(aug_block, "augmented_pair_source_scores_", [])
            ),
            "input_bin_edges": getattr(aug_block, "input_bin_edges_", {}),
            "source_observed_medians": dict(
                getattr(
                    aug_block,
                    "source_observed_medians_",
                    getattr(aug_block, "numeric_medians_", {}),
                )
            ),
            "kept_specs": list(getattr(aug_block, "kept_specs_", [])),
            "candidate_count": int(getattr(aug_block, "candidate_count_", 0)),
            "feature_names": list(getattr(aug_block, "feature_names_", [])),
        }
    # ────────────────────────────────────────────────────────────────────
    fit_state["interaction_relaxed_mining_state"] = {
        "interaction_relaxed_mining": bool(getattr(clf, "interaction_relaxed_mining", False)),
        "interaction_relaxed_feature_size": int(
            getattr(clf, "interaction_relaxed_feature_size", 10)
        ),
        "survivors": list(getattr(clf, "interaction_relaxed_mining_survivors_", [])),
    }
    # Columns identified at fit time as having at most one distinct observed
    # value. Excluded from native processing at both fit and predict time;
    # persisted so a loaded model continues excluding exactly the same
    # columns rather than treating them as ordinary columns again after a
    # save/load round trip. Absent on models saved before this field existed,
    # which load_model treats the same way fit() treats a model that hasn't
    # computed it yet: as an empty list, no columns excluded.
    fit_state["zero_variance_cols"] = list(getattr(clf, "_zero_variance_cols_", []) or [])
    members["clf_fit.json"] = _json_dumps(fit_state)

    # Patterns
    members["patterns.json"] = _json_dumps(
        [
            {"utility": float(pe.utility), "items": list(pe.items), "ig": float(pe.ig)}
            for pe in clf.patterns_
        ]
    )

    # Classifier numpy arrays
    clf_arrays: dict[str, np.ndarray] = {
        "classes_": clf.classes_,
        "cat_cols_mask_": clf.cat_cols_mask_.astype(np.bool_),
    }
    if hasattr(clf, "fallback_class_prior_"):
        clf_arrays["fallback_class_prior_"] = np.asarray(
            clf.fallback_class_prior_, dtype=np.float64
        )
    if hasattr(clf, "is_int_mask_") and clf.is_int_mask_ is not None:
        clf_arrays["is_int_mask_"] = clf.is_int_mask_.astype(np.bool_)
    if hasattr(clf, "_pattern_orders_"):
        clf_arrays["pattern_orders_"] = np.asarray(clf._pattern_orders_, dtype=np.int64)
    if hasattr(clf, "_interaction_pattern_mask_"):
        clf_arrays["interaction_pattern_mask_"] = np.asarray(
            clf._interaction_pattern_mask_, dtype=np.bool_
        )
    if hasattr(clf, "_original_scaler_"):
        scaler = clf._original_scaler_
        if hasattr(scaler, "mean_"):
            clf_arrays["original_scaler_mean_"] = np.asarray(scaler.mean_, dtype=np.float64)
        if hasattr(scaler, "scale_"):
            clf_arrays["original_scaler_scale_"] = np.asarray(scaler.scale_, dtype=np.float64)
        if hasattr(scaler, "var_"):
            clf_arrays["original_scaler_var_"] = np.asarray(scaler.var_, dtype=np.float64)
        if hasattr(scaler, "n_features_in_"):
            clf_arrays["original_scaler_n_features_in_"] = np.asarray(
                [scaler.n_features_in_], dtype=np.int64
            )
    if hasattr(clf, "_original_numeric_medians_"):
        clf_arrays["original_numeric_medians_"] = np.asarray(
            clf._original_numeric_medians_, dtype=np.float64
        )
    if (
        hasattr(clf, "_original_feature_mask_downstream_")
        and getattr(clf, "_original_feature_mask_downstream_", None) is not None
    ):
        clf_arrays["original_feature_mask_downstream_"] = np.asarray(
            clf._original_feature_mask_downstream_, dtype=np.bool_
        )
    if (
        hasattr(clf, "_original_feature_scores_downstream_")
        and getattr(clf, "_original_feature_scores_downstream_", None) is not None
    ):
        clf_arrays["original_feature_scores_downstream_"] = np.asarray(
            clf._original_feature_scores_downstream_, dtype=np.float64
        )
    if hasattr(clf, "_strict_topk_feature_mask_"):
        clf_arrays["strict_topk_feature_mask_"] = np.asarray(
            clf._strict_topk_feature_mask_, dtype=np.bool_
        )
    if hasattr(clf, "_strict_topk_feature_scores_"):
        clf_arrays["strict_topk_feature_scores_"] = np.asarray(
            clf._strict_topk_feature_scores_, dtype=np.float64
        )
    if hasattr(clf, "_downstream_pattern_support_"):
        clf_arrays["downstream_pattern_support_"] = np.asarray(
            clf._downstream_pattern_support_, dtype=np.float64
        )
    if hasattr(clf, "_downstream_non_missing_rate_"):
        clf_arrays["downstream_non_missing_rate_"] = np.asarray(
            clf._downstream_non_missing_rate_, dtype=np.float64
        )
    if hasattr(clf, "_downstream_variance_"):
        clf_arrays["downstream_variance_"] = np.asarray(clf._downstream_variance_, dtype=np.float64)
    aug_block = getattr(clf, "_augmented_pair_block_", None)
    if aug_block is not None:
        if hasattr(aug_block, "scaler_mean_"):
            clf_arrays["augmented_pair_scaler_mean_"] = np.asarray(
                aug_block.scaler_mean_, dtype=np.float64
            )
        if hasattr(aug_block, "scaler_scale_"):
            clf_arrays["augmented_pair_scaler_scale_"] = np.asarray(
                aug_block.scaler_scale_, dtype=np.float64
            )
        if hasattr(aug_block, "source_observed_medians_array_"):
            clf_arrays["augmented_pair_source_observed_medians_"] = np.asarray(
                aug_block.source_observed_medians_array_, dtype=np.float64
            )
        elif hasattr(aug_block, "numeric_medians_array_"):
            clf_arrays["augmented_pair_source_observed_medians_"] = np.asarray(
                aug_block.numeric_medians_array_, dtype=np.float64
            )
        if hasattr(aug_block, "pair_reference_values_"):
            clf_arrays["augmented_pair_reference_values_"] = np.asarray(
                aug_block.pair_reference_values_, dtype=np.float64
            )
        if hasattr(aug_block, "left_indices_"):
            clf_arrays["augmented_pair_left_indices_"] = np.asarray(
                aug_block.left_indices_, dtype=np.int64
            )
        if hasattr(aug_block, "right_indices_"):
            clf_arrays["augmented_pair_right_indices_"] = np.asarray(
                aug_block.right_indices_, dtype=np.int64
            )
        if hasattr(aug_block, "op_codes_"):
            clf_arrays["augmented_pair_op_codes_"] = np.asarray(aug_block.op_codes_, dtype=np.int8)
    members["arrays.npz"] = _npz_bytes(**clf_arrays)

    # TransactionDataWrapper
    td_config, td_arrays = _serialize_td(clf.td_)
    members["td_config.json"] = _json_dumps(td_config)
    members["td_arrays.npz"] = _npz_bytes(**td_arrays)

    # Downstream estimator
    est_config, est_arrays = _serialize_estimator(clf.model_)
    members["estimator.json"] = _json_dumps(est_config)
    members["estimator_arrays.npz"] = _npz_bytes(**est_arrays)

    # ── 2. Sign ───────────────────────────────────────────────────────────────
    key = _get_hmac_key()
    if key is not None:
        sig = _compute_archive_hmac(key, members)
    else:
        logger.warning(
            "HUGIML_MODEL_HMAC_KEY is not set; model will be saved without "
            "authentication.  Set this variable before loading models from "
            "untrusted sources."
        )
        sig = "0" * 64  # 32 zero bytes as hex

    members["hmac.sig"] = sig.encode()

    # ── 3. Write ZIP ──────────────────────────────────────────────────────────
    try:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, content in members.items():
                zf.writestr(name, content)
    except OSError as exc:
        raise HUGIMLSerializationError(f"Failed to write model to {path}: {exc}") from exc

    logger.debug("Saved v3 model artifact to %s (%d members)", path, len(members))


def load_model(
    path: str | os.PathLike,
    expected_type: type | None = None,
) -> Any:
    """Load a classifier from a file saved by :func:`save_model`.

    Supports:
    * v3  — ZIP/JSON/NumPy format (default since 2.1)
    * v1/v2 — legacy HMAC-pickle format (read-only; still authenticated)

    Parameters
    ----------
    path : str or Path
    expected_type : type, optional

    Returns
    -------
    HUGIMLClassifierNative

    Raises
    ------
    HUGIMLVersionError
        When schema version is incompatible.
    HUGIMLSerializationError
        When the file is corrupt, missing, has an invalid HMAC, or
        contains an unexpected type.
    """
    try:
        with open(path, "rb") as fh:
            magic_probe = fh.read(4)
    except OSError as exc:
        raise HUGIMLSerializationError(f"Failed to open model file {path}: {exc}") from exc

    if magic_probe == _LEGACY_MAGIC:
        logger.debug("Detected legacy v1/v2 pickle format for %s", path)
        clf = _load_legacy(path)
    elif magic_probe[:2] == b"PK":
        logger.debug("Detected v3 ZIP format for %s", path)
        clf = _load_v3(path)
    else:
        raise HUGIMLSerializationError(
            f"{path} is not a recognised HUG-IML model file "
            f"(magic bytes: {magic_probe!r}).  "
            "Expected a v3 ZIP archive or legacy HUGI-magic file."
        )

    if expected_type is not None and not isinstance(clf, expected_type):
        raise HUGIMLSerializationError(
            f"Deserialized object is {type(clf).__name__}, expected {expected_type.__name__}."
        )
    return clf


# ---------------------------------------------------------------------------
# v3 loader
# ---------------------------------------------------------------------------


def _load_v3(path: str | os.PathLike) -> Any:
    """Deserialize a v3 ZIP model archive."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            members: dict[str, bytes] = {name: zf.read(name) for name in zf.namelist()}
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise HUGIMLSerializationError(f"Failed to read ZIP archive {path}: {exc}") from exc

    required = {
        "manifest.json",
        "clf_init.json",
        "clf_fit.json",
        "patterns.json",
        "arrays.npz",
        "td_config.json",
        "td_arrays.npz",
        "estimator.json",
        "estimator_arrays.npz",
        "hmac.sig",
    }
    missing = required - set(members)
    if missing:
        raise HUGIMLSerializationError(
            f"Model archive {path} is incomplete; missing members: {sorted(missing)}"
        )

    manifest = json.loads(members["manifest.json"])
    schema_ver = manifest.get("schema_version", manifest.get("format_version", 0))
    if schema_ver < MIN_SCHEMA_VERSION:
        raise HUGIMLVersionError(
            f"Model schema version {schema_ver} is too old.  "
            f"Minimum supported: {MIN_SCHEMA_VERSION}.  Re-fit the model."
        )

    # Authenticate
    stored_sig = members["hmac.sig"].decode().strip()
    key = _get_hmac_key()
    if key is not None:
        if not _verify_archive_hmac(key, members, stored_sig):
            raise HUGIMLSerializationError(
                f"HMAC verification failed for {path}.  "
                "The file may have been tampered with or was saved with a different key."
            )
        logger.debug("HMAC verification passed for %s", path)
    elif _require_hmac():
        raise HUGIMLSerializationError(
            "HUGIML_REQUIRE_MODEL_HMAC is enabled but HUGIML_MODEL_HMAC_KEY is not "
            "configured.  Configure the key before loading production model files."
        )
    elif stored_sig != "0" * 64:
        logger.warning(
            "Model file %s contains an HMAC signature but HUGIML_MODEL_HMAC_KEY "
            "is not set; skipping authentication.  Configure the key to enable verification.",
            path,
        )

    def _load_npz(name: str) -> dict[str, np.ndarray]:
        buf = io.BytesIO(members[name])
        loaded = np.load(buf, allow_pickle=False)
        return dict(loaded)

    # ── Reconstruct the classifier ────────────────────────────────────────────
    from hugiml.classifier import FitMetadata, HUGIMLClassifierNative

    clf_init = json.loads(members["clf_init.json"])
    clf_fit = json.loads(members["clf_fit.json"])
    clf_arrays = _load_npz("arrays.npz")

    td_config = json.loads(members["td_config.json"])
    td_arrays = _load_npz("td_arrays.npz")

    est_config = json.loads(members["estimator.json"])
    est_arrays = _load_npz("estimator_arrays.npz")

    patterns_raw = json.loads(members["patterns.json"])

    # Build the classifier (unfitted shell)
    safe_init = {k: v for k, v in clf_init.items() if k != "base_estimator"}
    clf = HUGIMLClassifierNative(**safe_init)

    # Restore fitted attributes
    clf.n_features_in_ = int(clf_fit["n_features_in_"])
    clf.classes_ = clf_arrays["classes_"]
    clf.cat_cols_mask_ = clf_arrays["cat_cols_mask_"].astype(bool)
    if "is_int_mask_" in clf_arrays:
        clf.is_int_mask_ = clf_arrays["is_int_mask_"].astype(bool)
    else:
        clf.is_int_mask_ = None  # type: ignore[assignment]
    clf.feature_names_in_ = clf_fit.get("feature_names_in_")
    clf.binary_categorical_cols_ = list(clf_fit.get("binary_categorical_cols", []) or [])
    clf._native_available_ = clf_fit.get("_native_available_", False)
    if clf_fit.get("_degraded_reason"):
        clf._degraded_reason = clf_fit["_degraded_reason"]
    if clf_fit.get("n_categories_") is not None:
        clf.n_categories_ = clf_fit["n_categories_"]
    clf.feature_mode = clf_fit.get("feature_mode", getattr(clf, "feature_mode", "patterns_only"))
    init_execution_mode = clf_init.get("execution_mode", None)
    fit_has_execution_mode = "execution_mode" in clf_fit
    execution_mode = clf_fit.get("execution_mode", getattr(clf, "execution_mode", "audit"))
    if init_execution_mode is not None and init_execution_mode not in {"audit", "production"}:
        raise HUGIMLSerializationError(
            "Invalid execution_mode in clf_init.json: "
            f"{init_execution_mode!r}. Expected 'audit' or 'production'."
        )
    if (
        fit_has_execution_mode
        and init_execution_mode is not None
        and str(init_execution_mode) != str(execution_mode)
    ):
        raise HUGIMLSerializationError(
            "Inconsistent execution_mode between clf_init.json and clf_fit.json: "
            f"init={init_execution_mode!r}, fit={execution_mode!r}."
        )
    if execution_mode not in {"audit", "production"}:
        raise HUGIMLSerializationError(
            "Invalid execution_mode in model file: "
            f"{execution_mode!r}. Expected 'audit' or 'production'."
        )
    clf.execution_mode = execution_mode
    fallback_state = clf_fit.get("fallback_state") or {}
    clf.fallback_active_ = bool(fallback_state.get("active", False))
    if clf.fallback_active_:
        clf.fallback_strategy_ = str(fallback_state.get("strategy") or "constant_prior")
        clf.fallback_reason_ = fallback_state.get("reason")
        clf.fallback_n_samples_ = fallback_state.get("n_samples")
        clf.fallback_class_prior_ = clf_arrays.get(
            "fallback_class_prior_",
            np.full(len(clf.classes_), 1.0 / max(len(clf.classes_), 1), dtype=np.float64),
        )
        majority = fallback_state.get("majority_class")
        if majority is None and len(clf.classes_):
            majority = clf.classes_[int(np.argmax(clf.fallback_class_prior_))]
        clf.fallback_majority_class_ = majority
    if clf_fit.get("training_pattern_matrix_shape") is not None:
        clf._training_pattern_matrix_shape_ = tuple(
            int(v) for v in clf_fit.get("training_pattern_matrix_shape")
        )
    if clf_fit.get("training_pattern_matrix_nnz") is not None:
        clf._training_pattern_matrix_nnz_ = int(clf_fit.get("training_pattern_matrix_nnz"))
    if clf_fit.get("training_downstream_matrix_shape") is not None:
        clf._training_downstream_matrix_shape_ = tuple(
            int(v) for v in clf_fit.get("training_downstream_matrix_shape")
        )
    if clf_fit.get("training_downstream_matrix_nnz") is not None:
        clf._training_downstream_matrix_nnz_ = int(clf_fit.get("training_downstream_matrix_nnz"))
    clf._original_numeric_cols_ = clf_fit.get("original_numeric_cols", [])
    clf._original_cat_cols_ = clf_fit.get("original_cat_cols", [])
    clf._original_dummy_columns_ = clf_fit.get("original_dummy_columns", [])
    clf.topk_budget_strict = bool(
        clf_fit.get("topk_budget_strict", getattr(clf, "topk_budget_strict", False))
    )
    clf._downstream_feature_names_full_ = list(clf_fit.get("downstream_feature_names_full", []))
    clf._strict_topk_selected_feature_names_ = list(
        clf_fit.get("strict_topk_selected_feature_names", [])
    )
    clf._original_selected_feature_names_downstream_ = clf_fit.get(
        "original_selected_feature_names_downstream", None
    )
    if clf._original_selected_feature_names_downstream_ is not None:
        clf._original_selected_feature_names_downstream_ = list(
            clf._original_selected_feature_names_downstream_
        )
    clf._original_feature_names_downstream_full_ = clf_fit.get(
        "original_feature_names_downstream_full", None
    )
    if clf._original_feature_names_downstream_full_ is not None:
        clf._original_feature_names_downstream_full_ = list(
            clf._original_feature_names_downstream_full_
        )
    clf._strict_topk_applied_during_construction_ = bool(
        clf_fit.get("strict_topk_applied_during_construction", False)
    )
    clf._original_feature_mask_downstream_ = clf_arrays.get(
        "original_feature_mask_downstream_", None
    )
    if clf._original_feature_mask_downstream_ is not None:
        clf._original_feature_mask_downstream_ = clf._original_feature_mask_downstream_.astype(bool)
    clf._original_feature_scores_downstream_ = clf_arrays.get(
        "original_feature_scores_downstream_", np.zeros(0, dtype=np.float64)
    )
    clf._strict_topk_feature_mask_ = clf_arrays.get("strict_topk_feature_mask_", None)
    if clf._strict_topk_feature_mask_ is not None:
        clf._strict_topk_feature_mask_ = clf._strict_topk_feature_mask_.astype(bool)
    clf._strict_topk_feature_scores_ = clf_arrays.get(
        "strict_topk_feature_scores_", np.zeros(0, dtype=np.float64)
    )
    clf._downstream_pattern_support_ = clf_arrays.get("downstream_pattern_support_", None)
    clf._downstream_non_missing_rate_ = clf_arrays.get("downstream_non_missing_rate_", None)
    clf._downstream_variance_ = clf_arrays.get("downstream_variance_", None)
    clf._original_feature_names_downstream_ = clf_fit.get("original_feature_names_downstream", [])
    if "pattern_orders_" in clf_arrays:
        clf._pattern_orders_ = clf_arrays["pattern_orders_"].astype(int)
    if "interaction_pattern_mask_" in clf_arrays:
        clf._interaction_pattern_mask_ = clf_arrays["interaction_pattern_mask_"].astype(bool)
    if "original_numeric_medians_" in clf_arrays:
        clf._original_numeric_medians_ = pd.Series(
            clf_arrays["original_numeric_medians_"],
            index=clf._original_numeric_cols_,
            dtype=float,
        )
    if "original_scaler_mean_" in clf_arrays:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.mean_ = clf_arrays["original_scaler_mean_"]
        scaler.scale_ = clf_arrays.get("original_scaler_scale_", np.ones_like(scaler.mean_))
        scaler.var_ = clf_arrays.get("original_scaler_var_", scaler.scale_**2)
        scaler.n_features_in_ = int(
            clf_arrays.get("original_scaler_n_features_in_", np.array([len(scaler.mean_)]))[0]
        )
        clf._original_scaler_ = scaler

    # Patterns
    class _PE:
        __slots__ = ("utility", "items", "ig")

        def __init__(self, d: dict) -> None:
            self.utility = d["utility"]
            self.items = d["items"]
            self.ig = d["ig"]

    clf.patterns_ = [_PE(d) for d in patterns_raw]

    # FitMetadata
    if "fit_metadata_" in clf_fit:
        try:
            import dataclasses

            fm_data = clf_fit["fit_metadata_"]
            valid_fields = {f.name for f in dataclasses.fields(FitMetadata)}
            clf.fit_metadata_ = FitMetadata(
                **{k: v for k, v in fm_data.items() if k in valid_fields}
            )
        except Exception as exc:
            logger.debug("Could not restore FitMetadata: %s", exc, exc_info=True)

    # ── Missing value handling state ────────────────────────────────────
    missing_edges = clf_fit.get("missing_col_edges", {})
    if missing_edges is not None:
        clf._missing_col_edges_ = {
            name: np.array(edges, dtype=np.float64) for name, edges in missing_edges.items()
        }
    else:
        clf._missing_col_edges_ = {}
    # ─────────────────────────────────────────────────────────────────────
    # ── Adaptive binning state ──────────────────────────────────────────
    adap = clf_fit.get("adaptive_binning_state")
    if adap:
        clf._bin_edges_ = {
            name: np.array(edges, dtype=np.float64) for name, edges in adap["bin_edges"].items()
        }
        clf.per_feature_b_ = dict(adap["per_feature_b"])
        clf.ig_scores_ = {
            name: {int(b): float(v) for b, v in sc.items()}
            for name, sc in adap["ig_scores"].items()
        }
        # Rebuild the integer-code → original-label map that is derived from
        # _bin_edges_ but not stored separately in the .hugiml format.
        clf._rebuild_adaptive_code_label_map()
    else:
        clf._adaptive_code_label_map_ = {}
    # ────────────────────────────────────────────────────────────────────
    # ── Augmented pair transform state ──────────────────────────────────
    aug_state = clf_fit.get("augmented_pair_state") or {}
    clf.augmented_pair_transforms = bool(
        aug_state.get("augmented_pair_transforms", getattr(clf, "augmented_pair_transforms", True))
    )
    _legacy_max_features = aug_state.get("augmented_pair_max_features")
    clf.augmented_pair_mode = str(
        aug_state.get(
            "augmented_pair_mode", getattr(clf, "augmented_pair_mode", "interaction_information")
        )
    )
    clf.aug_feature_size = int(
        aug_state.get(
            "aug_feature_size",
            _legacy_max_features
            if _legacy_max_features is not None
            else getattr(clf, "aug_feature_size", 10),
        )
    )
    clf.ii_partner_size = aug_state.get("ii_partner_size", getattr(clf, "ii_partner_size", None))
    clf.max_pair_features = int(
        aug_state.get(
            "max_pair_features",
            _legacy_max_features
            if _legacy_max_features is not None
            else getattr(clf, "max_pair_features", 10),
        )
    )
    clf.augmented_pair_transforms_ = list(aug_state.get("transforms", []))
    clf.augmented_pair_selected_features_ = list(aug_state.get("selected_features", []))
    clf.augmented_pair_transforms_enabled_ = bool(aug_state.get("enabled", False))
    clf.augmented_pair_config_ = aug_state.get(
        "config", {"enabled": clf.augmented_pair_transforms_enabled_}
    )
    clf._augmented_pair_block_ = None
    block_state = aug_state.get("block_state")
    if block_state and clf.augmented_pair_transforms_enabled_:
        try:
            from hugiml.classifier import NativeAugmentedPairTransformBlock

            _legacy_block_max_features = block_state.get("max_features")
            block_aug_feature_size = int(
                block_state.get(
                    "aug_feature_size",
                    _legacy_block_max_features
                    if _legacy_block_max_features is not None
                    else clf.aug_feature_size,
                )
            )
            block_max_pair_features = int(
                block_state.get(
                    "max_pair_features",
                    _legacy_block_max_features
                    if _legacy_block_max_features is not None
                    else clf.max_pair_features,
                )
            )
            block = NativeAugmentedPairTransformBlock(
                augmented_pair_mode=str(
                    block_state.get("augmented_pair_mode", clf.augmented_pair_mode)
                ),
                aug_feature_size=block_aug_feature_size,
                max_pair_features=block_max_pair_features,
                ii_partner_size=block_state.get("ii_partner_size", clf.ii_partner_size),
                budget_topK=block_state.get("budget_topK"),
            )
            block.selected_aug_features_ = list(
                block_state.get("selected_aug_features", block_state.get("selected_ig_features", []))
            )
            block.selected_aug_scores_ = dict(
                block_state.get("selected_aug_scores", block_state.get("selected_ig_scores", {}))
            )
            block.augmented_pair_source_scores_ = list(
                block_state.get("augmented_pair_source_scores", [])
            )
            block.input_bin_edges_ = block_state.get("input_bin_edges", {})
            block.source_observed_medians_ = dict(
                block_state.get("source_observed_medians", block_state.get("numeric_medians", {}))
            )
            block.numeric_medians_ = dict(block.source_observed_medians_)
            block.kept_specs_ = list(block_state.get("kept_specs", []))
            block.candidate_count_ = int(block_state.get("candidate_count", 0))
            block.feature_names_ = list(block_state.get("feature_names", []))
            block.augmented_pair_transforms_ = list(clf.augmented_pair_transforms_)
            block.source_observed_medians_array_ = clf_arrays.get(
                "augmented_pair_source_observed_medians_",
                clf_arrays.get(
                    "augmented_pair_numeric_medians_",
                    np.asarray(
                        [
                            block.source_observed_medians_.get(c, 0.0)
                            for c in block.selected_aug_features_
                        ],
                        dtype=np.float64,
                    ),
                ),
            )
            block.numeric_medians_array_ = block.source_observed_medians_array_
            block.pair_reference_values_ = clf_arrays.get(
                "augmented_pair_reference_values_",
                np.asarray(
                    [float(spec.get("reference_raw_value", 0.0)) for spec in block.kept_specs_],
                    dtype=np.float64,
                ),
            )
            block.scaler_mean_ = clf_arrays.get(
                "augmented_pair_scaler_mean_", np.zeros(len(block.kept_specs_), dtype=np.float64)
            )
            block.scaler_scale_ = clf_arrays.get(
                "augmented_pair_scaler_scale_", np.ones(len(block.kept_specs_), dtype=np.float64)
            )
            block.left_indices_ = clf_arrays.get(
                "augmented_pair_left_indices_", np.zeros(len(block.kept_specs_), dtype=np.int64)
            )
            block.right_indices_ = clf_arrays.get(
                "augmented_pair_right_indices_", np.zeros(len(block.kept_specs_), dtype=np.int64)
            )
            block.op_codes_ = clf_arrays.get(
                "augmented_pair_op_codes_", np.zeros(len(block.kept_specs_), dtype=np.int8)
            )
            # Rebuild the public catalog from restored native arrays so
            # standardization metadata is available after load.
            try:
                block.augmented_pair_transforms_ = block._build_catalog()
                clf.augmented_pair_transforms_ = list(block.augmented_pair_transforms_)
            except Exception:
                block.augmented_pair_transforms_ = list(clf.augmented_pair_transforms_)
            block.augmented_pair_native_used_ = True
            clf._augmented_pair_block_ = block
        except Exception as exc:
            logger.debug("Could not restore augmented pair transforms: %s", exc, exc_info=True)
            clf._augmented_pair_block_ = None
    # ────────────────────────────────────────────────────────────────────
    # ── interaction_relaxed_mining state ────────────────────────────────
    relaxed_state = clf_fit.get("interaction_relaxed_mining_state") or {}
    clf.interaction_relaxed_mining = bool(
        relaxed_state.get("interaction_relaxed_mining", getattr(clf, "interaction_relaxed_mining", False))
    )
    clf.interaction_relaxed_feature_size = int(
        relaxed_state.get(
            "interaction_relaxed_feature_size", getattr(clf, "interaction_relaxed_feature_size", 10)
        )
    )
    if relaxed_state.get("survivors"):
        clf.interaction_relaxed_mining_survivors_ = list(relaxed_state["survivors"])
    # ────────────────────────────────────────────────────────────────────
    # ── zero-variance column exclusion state ────────────────────────────
    # Absent on archives written before this field existed; defaulting to
    # an empty list there means a model saved by an earlier version simply
    # excludes nothing extra after loading, identical to its own behaviour
    # before this field was introduced -- not a behaviour change for those
    # archives, just no opportunity to skip columns that version never
    # identified in the first place.
    clf._zero_variance_cols_ = list(clf_fit.get("zero_variance_cols") or [])
    # ────────────────────────────────────────────────────────────────────

    # TransactionDataWrapper
    clf.td_ = _deserialize_td(td_config, td_arrays)

    # Downstream estimator
    clf.model_ = _deserialize_estimator(est_config, est_arrays)

    # Threading lock (always fresh)
    import threading

    clf._fit_lock = threading.RLock()

    logger.debug(
        "Loaded HUGIML model from %s (%d patterns, schema_version=%d)",
        path,
        len(clf.patterns_),
        schema_ver,
    )
    return clf


# ---------------------------------------------------------------------------
# Legacy v1/v2 loader (restricted pickle, HMAC-authenticated)
# ---------------------------------------------------------------------------


def _load_legacy(path: str | os.PathLike) -> Any:
    """Load a model saved in the v1/v2 HMAC-pickle envelope format."""
    import pickle  # nosec B403 – legacy loader uses RestrictedUnpickler

    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
            if magic != _LEGACY_MAGIC:
                raise HUGIMLSerializationError(
                    f"{path} is not a valid legacy HUG-IML model file (magic bytes: {magic!r})."
                )
            (schema_ver,) = struct.unpack("<I", fh.read(4))
            if schema_ver < MIN_SCHEMA_VERSION:
                raise HUGIMLVersionError(
                    f"Model schema version {schema_ver} is too old.  "
                    f"Minimum supported: {MIN_SCHEMA_VERSION}.  Re-fit the model."
                )
            stored_tag = fh.read(_HMAC_LEN)
            payload = fh.read()
    except (OSError, struct.error) as exc:
        raise HUGIMLSerializationError(f"Failed to read legacy model {path}: {exc}") from exc

    version_bytes = struct.pack("<I", schema_ver)
    header_prefix = _LEGACY_MAGIC + version_bytes
    key = _get_hmac_key()

    if key is not None:
        expected_tag = hmac.new(key, header_prefix + payload, hashlib.sha256).digest()
        if not hmac.compare_digest(stored_tag, expected_tag):
            raise HUGIMLSerializationError(
                f"HMAC verification failed for {path}.  "
                "The file may have been tampered with or was saved with a different key."
            )
    elif _require_hmac():
        raise HUGIMLSerializationError(
            "HUGIML_REQUIRE_MODEL_HMAC is enabled but HUGIML_MODEL_HMAC_KEY is not configured."
        )
    elif stored_tag != b"\x00" * _HMAC_LEN:
        logger.warning(
            "Legacy model %s has an HMAC tag but HUGIML_MODEL_HMAC_KEY is not set; "
            "skipping authentication.",
            path,
        )

    try:
        return _safe_unpickle(payload)
    except pickle.UnpicklingError as exc:
        raise HUGIMLSerializationError(
            f"Restricted-pickle deserialization failed for {path}: {exc}"
        ) from exc


def _safe_unpickle(payload: bytes) -> Any:
    """Deserialize a pickle payload through a module- and type-restricting Unpickler."""
    import io
    import pickle  # nosec B403 – module-and-type-restricted Unpickler

    class _RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if any(module == m or module.startswith(m + ".") for m in _SAFE_MODULES):
                return super().find_class(module, name)
            if name in _SAFE_TYPES:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(
                f"Global '{module}.{name}' is not allowed during restricted deserialization."
            )

    return _RestrictedUnpickler(io.BytesIO(payload)).load()


# =============================================================================
# SBOM generation
# =============================================================================


def generate_sbom(output_path: str | None = None) -> dict[str, Any]:
    """Generate a Software Bill of Materials for the installed hugiml-core.

    Parameters
    ----------
    output_path : str, optional

    Returns
    -------
    dict  — CycloneDX-lite SBOM document.
    """
    import importlib.metadata as meta

    components = []
    for dep in ["numpy", "scipy", "scikit-learn", "pandas"]:
        try:
            version = meta.version(dep)
        except meta.PackageNotFoundError:
            try:
                version = meta.version(dep.replace("-", "_"))
            except meta.PackageNotFoundError:
                version = "unknown"
        components.append(
            {"name": dep, "version": version, "type": "library", "license": "BSD-3-Clause"}
        )

    sbom: dict[str, Any] = {
        "bomFormat": "CycloneDX-lite",
        "specVersion": "1.4",
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "component": {
                "name": "hugiml-core",
                "version": _hugiml_version(),
                "type": "library",
                "license": "Apache-2.0",
            },
        },
        "components": components,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(sbom, fh, indent=2)

    return sbom


def _hugiml_version() -> str:
    try:
        import importlib.metadata as meta

        return meta.version("hugiml-core")
    except Exception:
        from hugiml import __version__

        return __version__
