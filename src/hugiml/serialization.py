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

from hugiml.exceptions import HUGIMLSerializationError, HUGIMLVersionError

logger = logging.getLogger(__name__)

__all__ = [
    "MODEL_SCHEMA_VERSION",
    "MIN_SCHEMA_VERSION",
    "save_model",
    "load_model",
    "generate_sbom",
]

MODEL_SCHEMA_VERSION: int = 3
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
        "_TransactionDataWrapper",
        "FitMetadata",
        "PredictionMonitor",
        "DriftDetector",
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
        "_native_available_": bool(getattr(clf, "_native_available_", True)),
        "_degraded_reason": getattr(clf, "_degraded_reason", None),
        "n_categories_": getattr(clf, "n_categories_", None),
    }
    if hasattr(clf, "fit_metadata_") and clf.fit_metadata_ is not None:
        import dataclasses

        fit_state["fit_metadata_"] = dataclasses.asdict(clf.fit_metadata_)
    # ── v1.1.0 missing value handling state ──────────────────────────────
    # _missing_col_edges_ stores quantile edges for columns that had NaN/Inf
    # in training data.  Serialised as float lists for JSON compatibility.
    if getattr(clf, "_missing_col_edges_", None):
        fit_state["missing_col_edges"] = {
            name: edges.tolist()
            for name, edges in clf._missing_col_edges_.items()
        }
    # ─────────────────────────────────────────────────────────────────────
    # ── v1.1.0 adaptive binning state ────────────────────────────────────
    # _bin_edges_ (dict[str, np.ndarray]) is serialised as JSON-compatible
    # lists.  per_feature_b_ and ig_scores_ are plain dicts.  JSON object
    # keys must be strings so ig_scores_ int keys are stringified here and
    # converted back to int in load_model.
    if getattr(clf, "adaptive_binning", False) and getattr(clf, "_bin_edges_", None):
        fit_state["adaptive_binning_state"] = {
            "bin_edges": {
                name: edges.tolist()
                for name, edges in clf._bin_edges_.items()
            },
            "per_feature_b": dict(clf.per_feature_b_),
            "ig_scores": {
                name: {str(b): float(v) for b, v in sc.items()}
                for name, sc in clf.ig_scores_.items()
            },
        }
    # ────────────────────────────────────────────────────────────────────
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
    if hasattr(clf, "is_int_mask_") and clf.is_int_mask_ is not None:
        clf_arrays["is_int_mask_"] = clf.is_int_mask_.astype(np.bool_)
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
    clf._native_available_ = clf_fit.get("_native_available_", False)
    if clf_fit.get("_degraded_reason"):
        clf._degraded_reason = clf_fit["_degraded_reason"]
    if clf_fit.get("n_categories_") is not None:
        clf.n_categories_ = clf_fit["n_categories_"]

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

    # ── v1.1.0 missing value handling state ──────────────────────────────
    missing_edges = clf_fit.get("missing_col_edges")
    if missing_edges:
        clf._missing_col_edges_ = {
            name: np.array(edges, dtype=np.float64)
            for name, edges in missing_edges.items()
        }
    else:
        clf._missing_col_edges_ = {}
    # ─────────────────────────────────────────────────────────────────────
    # ── v1.1.0 adaptive binning state ────────────────────────────────────
    adap = clf_fit.get("adaptive_binning_state")
    if adap:
        clf._bin_edges_ = {
            name: np.array(edges, dtype=np.float64)
            for name, edges in adap["bin_edges"].items()
        }
        clf.per_feature_b_ = dict(adap["per_feature_b"])
        clf.ig_scores_ = {
            name: {int(b): float(v) for b, v in sc.items()}
            for name, sc in adap["ig_scores"].items()
        }
    # ────────────────────────────────────────────────────────────────────

    # TransactionDataWrapper
    clf.td_ = _deserialize_td(td_config, td_arrays)

    # Downstream estimator
    clf.model_ = _deserialize_estimator(est_config, est_arrays)

    # Threading lock (always fresh)
    import threading

    clf._fit_lock = threading.RLock()

    logger.debug(
        "Loaded v3 model from %s (%d patterns, schema_version=%d)",
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
