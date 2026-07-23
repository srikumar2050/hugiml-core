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


"""Classifier tuning and cached grid evaluation."""

from __future__ import annotations

import copy
import dataclasses
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline

from hugiml._classifier_runtime import _CORE_AVAILABLE, _core
from hugiml._classifier_support import _wire_hugiml_feature_metadata
from hugiml.exceptions import HUGIMLParamError, HUGIMLValidationError
from hugiml.hyperparameter_configs import get_hugiml_grid

HUGIMLClassifier = Any


def _hugiml_auc_score_for_fast_grid(y_true: Any, proba: np.ndarray, classes: np.ndarray) -> float:
    """Internal validation AUC scorer used by fast_grid_tune()."""
    from sklearn.metrics import roc_auc_score

    y_arr = np.asarray(y_true)
    if proba.ndim != 2:
        raise HUGIMLValidationError("predict_proba must return a 2D array.")
    if proba.shape[1] == 2:
        return float(roc_auc_score(y_arr, proba[:, 1]))
    return float(roc_auc_score(y_arr, proba, multi_class="ovr", average="macro"))


def _hugiml_expand_grid_for_fast_tune(
    param_grid: dict[str, list] | str | None,
) -> list[dict[str, Any]]:
    """Expand a compact sklearn-style parameter grid for fast HUGIML tuning.

    ``param_grid`` may be a dict of parameter lists (the sklearn-style grid
    itself), the name of one of the grids in ``hugiml.hyperparameter_configs``
    (for example ``"interpretability"``), or ``None`` to use the default
    named grid.
    """
    from itertools import product

    if param_grid is None or isinstance(param_grid, str):
        grid = get_hugiml_grid(param_grid)
    else:
        grid = param_grid
    if not isinstance(grid, dict) or not grid:
        raise HUGIMLParamError("param_grid must be a non-empty dict of parameter lists.")
    keys = list(grid.keys())
    values = []
    for key in keys:
        val = grid[key]
        if isinstance(val, (str, bytes)) or not hasattr(val, "__iter__"):
            val = [val]
        val = list(val)
        if not val:
            raise HUGIMLParamError(f"param_grid[{key!r}] must contain at least one value.")
        values.append(val)
    return [dict(zip(keys, vals)) for vals in product(*values)]


def _hugiml_validate_fast_tune_grid(candidates: list[dict[str, Any]]) -> dict[str, list]:
    """Validate that a grid is safe for exact cached tuning.

    The fast path is exact when adaptive binning is enabled and only mining/
    representation dimensions (G, L, topK, feature_mode) plus the downstream
    estimator choice (base_estimator) vary. Because G is part of the native
    mining call, candidates are cached in separate constant-G groups. B may
    appear and even vary, but is ignored while adaptive_binning=True because
    per-feature binning supplies the effective discretisation.

    base_estimator is safe to allow here even though it isn't a mining
    parameter: it only affects the final "fit an estimator on the already-
    built downstream feature matrix" step (see _make_estimator), never
    mining, pattern selection, or downstream-matrix construction -- see
    _hugiml_prepare_downstream_template_from_cached_base /
    _hugiml_fit_downstream_estimator_from_template, which split those two
    concerns precisely so a base_estimator-varying grid (e.g.
    hyperparameter_configs.py's performance_ho) can still reuse the mining
    and downstream-matrix cache across its base_estimator candidates.
    """
    if not candidates:
        raise HUGIMLParamError("No grid candidates supplied.")

    varying = {
        key
        for key in set().union(*(set(c.keys()) for c in candidates))
        if len({repr(c.get(key, None)) for c in candidates}) > 1
    }
    allowed_varying = {"B", "G", "L", "topK", "feature_mode", "base_estimator"}
    disallowed = sorted(varying - allowed_varying)
    if disallowed:
        raise HUGIMLParamError(
            "fast_grid_tune requires only G, L, topK, feature_mode, and base_estimator "
            f"to vary (B is ignored under adaptive_binning=True). Varying unsupported keys: {disallowed}."
        )

    adaptive_values = {bool(c.get("adaptive_binning", True)) for c in candidates}
    if adaptive_values != {True}:
        raise HUGIMLParamError("fast_grid_tune requires adaptive_binning=True for every candidate.")

    g_values = sorted({float(c.get("G", 1e-2)) for c in candidates})
    L_values = sorted({int(c.get("L", 1)) for c in candidates})
    topk_values = sorted({int(c.get("topK", 30)) for c in candidates})
    feature_modes = sorted({str(c.get("feature_mode", "patterns_only")) for c in candidates})
    if any(k <= 0 for k in topk_values):
        raise HUGIMLParamError(
            "fast_grid_tune currently supports positive integer topK values only."
        )
    if any(L < 1 for L in L_values):
        raise HUGIMLParamError("fast_grid_tune currently supports L >= 1 only.")

    allowed_modes = {"patterns_only", "original_plus_patterns", "original_plus_interactions"}
    bad_modes = sorted(set(feature_modes) - allowed_modes)
    if bad_modes:
        raise HUGIMLParamError(f"Unsupported feature_mode values for fast_grid_tune: {bad_modes}.")

    # Deduplicated by repr(): distinct base_estimator *configurations*, not
    # distinct Python object identities -- two None-vs-RPTE candidates
    # collapse to the same 2 entries here regardless of how many times
    # each was repeated across an outer grid expansion.
    seen_reprs: set[str] = set()
    base_estimator_values: list[Any] = []
    for c in candidates:
        value = c.get("base_estimator", None)
        key = repr(value)
        if key not in seen_reprs:
            seen_reprs.add(key)
            base_estimator_values.append(value)

    return {
        "L_values": L_values,
        "topK_values": topk_values,
        "feature_modes": feature_modes,
        "G_values": g_values,
        "base_estimator_values": base_estimator_values,
        # Backward-compatible key used by older callers/tests.
        "G": g_values,
    }


def _hugiml_shallow_candidate_from_base(base: HUGIMLClassifier) -> HUGIMLClassifier:
    """Create a candidate that shares immutable cached mining artefacts with base."""
    cand = base.__class__(**base.get_params(deep=False))
    share_attrs = [
        "cat_cols_mask_",
        "is_int_mask_",
        "feature_names_in_",
        "_bin_edges_",
        "_missing_col_edges_",
        "_adaptive_code_label_map_",
        "_adaptive_precoded_features_",
        "per_feature_b_",
        "ig_scores_",
        "td_",
        "raw_patterns_",
        "classes_",
        "n_features_in_",
        "_native_available_",
    ]
    for attr in share_attrs:
        if hasattr(base, attr):
            setattr(cand, attr, getattr(base, attr))
    return cand


def _hugiml_prepare_downstream_template_from_cached_base(
    base: HUGIMLClassifier,
    X_train_original: Any,
    y_train: Any,
    L_value: int,
    topK_value: int,
    feature_mode: str,
    execution_mode: str = "audit",
    native_augmented_pair_cache: Any | None = None,
    native_augmented_pair_score_topK: int | None = None,
) -> HUGIMLClassifier:
    """Build one (G, L, topK, feature_mode) candidate's downstream feature
    matrix from a max-topK cached mining base -- everything a candidate
    needs before an estimator is fit against it: pattern selection,
    augmented-pair transforms, and the downstream matrix itself
    (``x_train_downstream_``). Deliberately stops short of building or
    fitting ``model_``: that step is the only one that depends on
    ``base_estimator``, so keeping it separate lets
    ``_hugiml_fit_downstream_estimator_from_template`` fit several
    base_estimator candidates against the same prepared template without
    repeating mining or downstream-matrix construction for each one (see
    hyperparameter_configs.py's performance_ho grid, which varies
    base_estimator between HUGIML's built-in logistic regression and RPTE).

    A returned template that already has ``model_`` set (the constant-
    prior fallback for a zero-pattern candidate) is already a complete,
    usable model on its own -- there is no downstream feature matrix for
    base_estimator to act on, so every base_estimator candidate sharing
    this template is equivalent and
    _hugiml_fit_downstream_estimator_from_template returns it unchanged.
    """
    cand = _hugiml_shallow_candidate_from_base(base)
    cand.L = int(L_value)
    cand.topK = int(topK_value)
    cand.feature_mode = str(feature_mode)
    cand.execution_mode = str(execution_mode)

    raw_patterns = list(getattr(base, "raw_patterns_", []))[: int(topK_value)]
    n_train = len(y_train)
    base_hup = getattr(base, "x_train_hup_", None)
    if base_hup is None:
        raise RuntimeError(
            "fast_grid_tune requires cached training pattern matrices. "
            "The cache model was created in production mode or otherwise does not retain "
            "x_train_hup_; run tuning with execution_mode='audit'."
        )
    if int(L_value) == 1:
        cand.patterns_ = raw_patterns
        # Fused L=1 path returns columns in raw_patterns_ order; slicing is exact.
        cand.x_train_hup_ = base_hup[:, : len(cand.patterns_)]
    else:
        native_td = getattr(getattr(base, "td_", None), "_td", getattr(base, "td_", None))
        old_td = getattr(cand, "td_", None)
        cand.td_ = native_td
        cand.patterns_, cached_coo = cand._deduplicate_patterns_by_coverage(raw_patterns, n_train)
        cand.td_ = old_td
        if cached_coo is not None:
            rows, cols = cached_coo
        elif len(cand.patterns_) > 0:
            rows, cols = _core.build_train_matrix(native_td, cand.patterns_)
        else:
            rows = cols = np.zeros(0, dtype=np.int32)
        data = np.ones(len(rows), dtype=np.float32)
        cand.x_train_hup_ = csr_matrix(
            (data, (rows, cols)), shape=(n_train, len(cand.patterns_)), dtype=np.float32
        )

    if len(cand.patterns_) == 0 and cand.feature_mode == "patterns_only":
        cand._setup_constant_prior_fallback(
            y_train,
            reason=(
                "No HUG patterns found for cached candidate. Try reducing G, increasing topK, "
                "or using original_plus_patterns."
            ),
        )
        cand._apply_execution_mode_retention()
        return cand

    cand._setup_feature_mode_metadata()
    cand._native_augmented_pair_cache_ = native_augmented_pair_cache
    cand._native_augmented_pair_score_topK_ = native_augmented_pair_score_topK
    try:
        cand._setup_augmented_pair_transforms(X_train_original, y_train, fit=True)
    finally:
        if hasattr(cand, "_native_augmented_pair_cache_"):
            delattr(cand, "_native_augmented_pair_cache_")
        if hasattr(cand, "_native_augmented_pair_score_topK_"):
            delattr(cand, "_native_augmented_pair_score_topK_")
    cand._current_y_for_downstream_topk_ = y_train
    try:
        X_down = cand._make_downstream_features(X_train_original, cand.x_train_hup_, fit=True)
    finally:
        if hasattr(cand, "_current_y_for_downstream_topk_"):
            delattr(cand, "_current_y_for_downstream_topk_")
    X_down = cand._apply_strict_topk_budget_fit(X_down, y_train)
    cand.x_train_downstream_ = X_down
    cand._cache_downstream_feature_metadata()
    # Intentionally does not build/fit model_ here -- see
    # _hugiml_fit_downstream_estimator_from_template.
    return cand


def _hugiml_fit_downstream_estimator_from_template(
    template: HUGIMLClassifier,
    base_estimator_value: Any,
    y_train: Any,
) -> HUGIMLClassifier:
    """Fit one base_estimator candidate's downstream estimator against an
    already-prepared template's cached downstream feature matrix, without
    repeating mining, pattern selection, or downstream-matrix construction
    -- see _hugiml_prepare_downstream_template_from_cached_base.

    A shallow copy of `template` is used per call (numpy/scipy arrays are
    shared by reference, not duplicated -- safe since nothing here mutates
    them) so that fitting one base_estimator candidate never affects
    another candidate built from the same template.
    """
    if hasattr(template, "model_"):
        # Degenerate (constant-prior) template: no downstream feature
        # matrix exists for base_estimator to act on, so every candidate
        # sharing this template is identical regardless of base_estimator.
        return template

    cand = copy.copy(template)
    cand.base_estimator = base_estimator_value
    X_down = cand.x_train_downstream_
    cand.model_ = Pipeline([("clf", cand._make_estimator(len(cand.classes_)))])
    _downstream_names_for_wiring = cand._get_downstream_feature_names()
    if len(_downstream_names_for_wiring) != X_down.shape[1]:
        raise RuntimeError(
            f"Internal error: downstream feature name count "
            f"({len(_downstream_names_for_wiring)}) does not match the downstream "
            f"matrix width ({X_down.shape[1]}). Refusing to wire HUGIML feature "
            f"metadata into the downstream estimator with mismatched names -- every "
            f"later column would be misattributed."
        )
    _wire_hugiml_feature_metadata(
        cand.model_.named_steps["clf"],
        _downstream_names_for_wiring,
        cand.get_augmented_pair_transforms(),
        cand.get_pattern_provenance(),
        cand.get_original_feature_standardization(),
    )
    cand.model_.fit(X_down, y_train)
    cand._apply_execution_mode_retention()
    # Intentionally avoid drift baseline and rich metadata during tuning. The
    # returned best_model is immediately usable for prediction; call fit() on the
    # selected params if full production metadata/drift baseline is required.
    return cand


def _hugiml_prepare_candidate_from_cached_base(
    base: HUGIMLClassifier,
    X_train_original: Any,
    y_train: Any,
    L_value: int,
    topK_value: int,
    feature_mode: str,
    execution_mode: str = "audit",
    native_augmented_pair_cache: Any | None = None,
    native_augmented_pair_score_topK: int | None = None,
) -> HUGIMLClassifier:
    """Build and fit one exact candidate from a max-topK cached base model.

    Kept as a single-call convenience wrapper composing
    _hugiml_prepare_downstream_template_from_cached_base and
    _hugiml_fit_downstream_estimator_from_template (using the base's own
    base_estimator) for any caller that doesn't need to fit several
    base_estimator candidates against the same template -- see those two
    functions' docstrings for why the split exists.
    """
    template = _hugiml_prepare_downstream_template_from_cached_base(
        base,
        X_train_original,
        y_train,
        L_value,
        topK_value,
        feature_mode,
        execution_mode,
        native_augmented_pair_cache,
        native_augmented_pair_score_topK,
    )
    return _hugiml_fit_downstream_estimator_from_template(template, base.base_estimator, y_train)


def _hugiml_build_fast_tune_adaptive_context(
    cls,
    X_train: Any,
    y_train: Any,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a fold-local adaptive pre-binning context for fast_grid_tune.

    This cache is exact only when adaptive binning is ordinary per-feature
    adaptive binning.  It is intentionally disabled for interaction-relaxed
    pair-aware adaptive binning, where L>1 can change the selected adaptive
    pairs.  The returned object is passed only to internal cache-building fits
    and is not attached to any fitted model returned to users.
    """
    if not bool(params.get("adaptive_binning", True)):
        return None
    if bool(params.get("interaction_relaxed_mining", False)):
        return None

    builder = cls(**params)
    builder._validate_params()
    builder._resolve_col_meta(X_train)
    y_arr = cls._safe_cast_y(y_train)
    if _CORE_AVAILABLE and hasattr(_core, "select_adaptive_bins"):
        X_pre = builder._apply_adaptive_binning_cpp(X_train, y_arr)
    else:
        X_pre = builder._apply_adaptive_binning(X_train, y_arr)

    attrs: dict[str, Any] = {}
    for name in (
        "cat_cols_mask_",
        "is_int_mask_",
        "feature_names_in_",
        "binary_categorical_cols_",
        "_bin_edges_",
        "_missing_col_edges_",
        "_adaptive_code_label_map_",
        "_adaptive_precoded_features_",
        "per_feature_b_",
        "ig_scores_",
    ):
        if hasattr(builder, name):
            value = getattr(builder, name)
            if isinstance(value, dict):
                value = dict(value)
            elif isinstance(value, set):
                value = set(value)
            elif isinstance(value, list):
                value = list(value)
            elif isinstance(value, np.ndarray):
                value = value.copy()
            attrs[name] = value
    return {"X_pre": X_pre, "attrs": attrs}


def _hugiml_fast_grid_tune(
    cls,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    param_grid: dict[str, list] | str | None = None,
    *,
    base_params: dict[str, Any] | None = None,
    scoring: str = "roc_auc",
    refit_full: bool = False,
    return_results: bool = True,
) -> dict[str, Any]:
    """Exact cached tuner for the compact adaptive HUGIML grid.

    Requirements
    ------------
    - adaptive_binning=True for every candidate.
    - G may vary; the tuner partitions candidates into constant-G cache groups.
    - Only G, L, topK, and feature_mode vary. B may appear in the grid but is
      ignored for cache partitioning because adaptive binning chooses per-feature
      bins and fit() passes sentinel B=2 to the native transaction builder.
    - max_fit_seconds and max_mining_seconds must be None to guarantee
      equivalence to the ordinary grid loop; timeout/degradation can make
      cached mining fits differ from standalone candidates.

    ``base_params['execution_mode']`` defaults to ``'production'`` when not
    supplied, since a tuning sweep evaluates many candidates and the
    line-level allocation diagnostics that ``'audit'`` provides are rarely
    useful for any individual one of them; pass
    ``base_params={'execution_mode': 'audit', ...}`` to opt back in. This
    only affects line-level tracing cost and the post-fit retention of
    training matrices -- it does not change which patterns are mined,
    scores, or rankings, all of which are identical either way (the
    per-(G, L, topK) cache fits that do the actual mining are tagged
    'audit' internally regardless, so that the candidates derived from
    them keep access to the cached pattern matrix; see the cache-building
    loop below).

    With ``refit_full=False`` (the default), ``best_model`` is one of these
    lightweight per-candidate objects: drift-baseline and rich metadata were
    never computed for it regardless of execution_mode (see below), and
    under the 'production' default its training pattern matrix is not
    retained either, so get_pattern_info() also raises on it -- call this
    function again with ``refit_full=True``, or call fit() directly with the
    selected params, for a model meant to be inspected afterward rather than
    just used for prediction.

    With ``refit_full=True``, the refit that produces the returned
    ``best_model`` defaults to ``execution_mode='audit'`` regardless of what
    the search candidates used, unless the caller's own ``base_params``
    explicitly named an ``execution_mode`` -- the search benefits from
    'production' for speed across many candidates, but the one model
    actually handed back for the caller to keep and inspect defaults to
    having full access to get_pattern_info(), detect_drift(),
    get_drift_psi(), and feature_importances() without missing-artifact
    warnings, rather than inheriting the search's speed-oriented default.

    Returns a dict with best_model, best_params, best_score, cv_results, and
    cache timings. Uses the same scorer as the ordinary grid path for all
    supported scoring values. During tuning it skips drift-baseline and rich final
    metadata; set refit_full=True to refit the selected model with normal fit().
    """
    t_start = time.perf_counter()
    candidates = _hugiml_expand_grid_for_fast_tune(param_grid)
    grid_info = _hugiml_validate_fast_tune_grid(candidates)
    params0 = dict(base_params or {})
    params0.setdefault("adaptive_binning", True)
    params0.setdefault("use_hotpath", True)
    # Recorded before the setdefault below so refit_full can tell "the caller
    # asked for production" apart from "this default applies to every search
    # candidate but a refit_full model should still get full audit access by
    # default" -- see the refit_full block further down.
    _caller_set_execution_mode = "execution_mode" in params0
    params0.setdefault("execution_mode", "production")
    # Do not set a single global G here; G is part of mining and is constant per
    # cache group below.  A caller-supplied base G is used only for candidates
    # that omit G from the grid.
    params0.setdefault("G", grid_info["G_values"][0])
    # Any grid key beyond the ones explicitly varying-and-handled below
    # (L, topK, feature_mode, G, base_estimator) is required by
    # _hugiml_validate_fast_tune_grid to be constant across every
    # candidate -- but that constant value was never actually threaded
    # into the cached mining base's own params here, only whatever the
    # caller's base_params happened to set. Invisible for a grid that
    # only ever varies the core four, but silently wrong for any grid
    # (e.g. interpretability_ho, which explicitly sets
    # convert_binary_to_categorical=True) that also sets another parameter
    # mining itself depends
    # on: the cached base would mine under the class default instead of
    # the grid's actual value. Fold each such constant into params0
    # (unless the caller's base_params already set it explicitly, which
    # takes precedence) so the cached mining base -- and everything
    # derived from it via _hugiml_shallow_candidate_from_base -- agrees
    # with what every candidate in the grid actually requested.
    _explicit_keys = {"L", "topK", "feature_mode", "G", "base_estimator", "adaptive_binning"}
    _grid_keys: set[str] = set()
    for _c in candidates:
        _grid_keys.update(_c.keys())
    for _key in sorted(_grid_keys - _explicit_keys):
        if _key not in params0:
            params0[_key] = candidates[0][_key]
    if (
        params0.get("max_fit_seconds", None) is not None
        or params0.get("max_mining_seconds", None) is not None
    ):
        raise HUGIMLParamError(
            "fast_grid_tune requires max_fit_seconds=None and max_mining_seconds=None "
            "for exact equivalence."
        )

    requested_execution_mode = str(params0.get("execution_mode", "production"))
    if requested_execution_mode not in {"audit", "production"}:
        raise HUGIMLParamError(
            "execution_mode must be either 'audit' or 'production'. "
            f"Got {requested_execution_mode!r}."
        )
    y_train_arr = cls._safe_cast_y(y_train)
    y_val_arr = np.asarray(y_val)
    X_train_original = cls(**params0)._copy_input_for_downstream(X_train)

    disable_prep_cache = os.environ.get("HUGIML_FAST_TUNE_DISABLE_PREP_CACHE", "0") == "1"
    disable_validation_cache = (
        os.environ.get("HUGIML_FAST_TUNE_DISABLE_VALIDATION_CACHE", "0") == "1"
    )
    t_adapt_ctx = time.perf_counter()
    adaptive_context = None
    if not disable_prep_cache and not bool(params0.get("interaction_relaxed_mining", False)):
        # Lazy fold-local context. The first eligible L>1 cache fit performs
        # ordinary adaptive binning and stores its exact pre-binned output;
        # later L>1 cache fits reuse it. This avoids an extra up-front
        # adaptive pass and keeps L=1 on the native fused adaptive path.
        adaptive_context = {"misses": 0}
    adaptive_context_seconds = time.perf_counter() - t_adapt_ctx
    transaction_cache: dict[Any, Any] = {}

    # Correctness note: topK is NOT derived by mining max(topK) once and slicing.
    # Empirically, the native miner can return additional valid patterns when a
    # larger topK is requested, so a smaller standalone topK run is not always
    # equivalent to a prefix of the larger run.  To guarantee identical validation
    # scores to the ordinary grid loop, cache one mining fit per (G, L, topK)
    # group and reuse that cache only across feature_mode candidates.  Within
    # each cache fit, fit() already sorts raw_patterns_ by descending utility with
    # tuple(items) tie-breaking before downstream construction.
    base_by_G_L_topK: dict[tuple[float, int, int], HUGIMLClassifier] = {}
    cache_fit_seconds: dict[str, float] = {}
    needed_cache_keys = sorted(
        {
            (
                float(c.get("G", params0.get("G", 1e-2))),
                int(c.get("L", 1)),
                int(c.get("topK", 30)),
            )
            for c in candidates
        }
    )
    for G_value, L_value, topK_value in needed_cache_keys:
        base_fit_params = dict(params0)
        base_fit_params.update(
            {
                "adaptive_binning": True,
                "L": int(L_value),
                "topK": int(topK_value),
                # Use the richest ordinary mode so raw input is preserved and
                # empty-pattern fallbacks do not fail while building the cache.
                "feature_mode": "original_plus_patterns",
                "G": float(G_value),
                # Cached tuning needs training matrices.  Even if callers pass
                # production in base_params, the internal cache must retain audit
                # artifacts; final refit below can still use caller params.
                "execution_mode": "audit",
            }
        )
        # B may be present in the original grid, but it is intentionally ignored
        # under adaptive_binning=True.
        t_fit = time.perf_counter()
        base = cls(**base_fit_params)
        base._fast_tune_cache_only = True
        # Reuse pre-binning and transaction preparation for the non-fused L>1
        # cache fits.  L=1 keeps the native fused adaptive path unchanged so
        # ordinary fast_grid_tune remains bit-for-bit aligned with base 1.1.15.
        if adaptive_context is not None and int(L_value) > 1:
            base._fast_tune_adaptive_context = adaptive_context
            base._fast_tune_transaction_cache = transaction_cache
        try:
            base.fit(X_train, y_train_arr)
        finally:
            base.__dict__.pop("_fast_tune_cache_only", None)
            base.__dict__.pop("_fast_tune_adaptive_context", None)
            base.__dict__.pop("_fast_tune_transaction_cache", None)
        cache_fit_seconds[f"G={float(G_value):.12g},L={int(L_value)},topK={int(topK_value)}"] = (
            time.perf_counter() - t_fit
        )
        base_by_G_L_topK[(float(G_value), int(L_value), int(topK_value))] = base

    native_augmented_pair_cache = (
        _core.AugmentedPairCache()
        if _CORE_AVAILABLE and hasattr(_core, "AugmentedPairCache")
        else None
    )
    native_augmented_pair_score_topK = max(
        [int(c.get("topK", params0.get("topK", 30))) for c in candidates if int(c.get("L", 1)) > 1]
        or [0]
    )

    validation_cache: dict[Any, Any] | None = None if disable_validation_cache else {}
    # Second-level cache: one prepared downstream feature matrix per (G, L,
    # topK, feature_mode), shared across every base_estimator candidate at
    # that key (see _hugiml_prepare_downstream_template_from_cached_base /
    # _hugiml_fit_downstream_estimator_from_template). For a grid that
    # doesn't vary base_estimator this costs nothing extra: each key is
    # still built exactly once, same as before the split existed.
    downstream_template_cache: dict[tuple[float, int, int, str], HUGIMLClassifier] = {}

    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_model: HUGIMLClassifier | None = None
    best_params: dict[str, Any] | None = None

    for candidate_params in candidates:
        L_value = int(candidate_params.get("L", 1))
        topK_value = int(candidate_params.get("topK", 30))
        feature_mode = str(candidate_params.get("feature_mode", "patterns_only"))
        G_value = float(candidate_params.get("G", params0.get("G", 1e-2)))
        base_estimator_value = candidate_params.get("base_estimator", params0.get("base_estimator"))
        t_cand = time.perf_counter()
        status = "ok"
        err = None
        score = np.nan
        model = None
        try:
            template_key = (G_value, L_value, topK_value, feature_mode)
            template = downstream_template_cache.get(template_key)
            if template is None:
                template = _hugiml_prepare_downstream_template_from_cached_base(
                    base_by_G_L_topK[(G_value, L_value, topK_value)],
                    X_train_original,
                    y_train_arr,
                    L_value,
                    topK_value,
                    feature_mode,
                    requested_execution_mode,
                    native_augmented_pair_cache,
                    native_augmented_pair_score_topK,
                )
                downstream_template_cache[template_key] = template
            model = _hugiml_fit_downstream_estimator_from_template(
                template, base_estimator_value, y_train_arr
            )
            score = _hugiml_score_model_for_tune(model, X_val, y_val_arr, scoring, validation_cache)
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                best_model = model
                best_params = dict(candidate_params)
                best_params["adaptive_binning"] = True
                best_params["G"] = G_value
                best_params.setdefault("execution_mode", requested_execution_mode)
        except (
            Exception
        ) as exc:  # keep failed candidates visible like GridSearchCV error_score=np.nan
            status = "failed"
            err = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "params": dict(candidate_params),
                "L": L_value,
                "topK": topK_value,
                "feature_mode": feature_mode,
                "G": G_value,
                "mean_test_score": score,
                "status": status,
                "error": err,
                "elapsed_seconds": time.perf_counter() - t_cand,
            }
        )

    if best_model is None or best_params is None:
        raise HUGIMLValidationError("All fast_grid_tune candidates failed.")

    if refit_full:
        refit_params = dict(params0)
        refit_params.update(best_params)
        # The search above runs every candidate under 'production' by default
        # for speed; refit_full's purpose is producing one normal, fully
        # inspectable model from the winning configuration, so unless the
        # caller explicitly asked for 'production', this refit uses 'audit'
        # so get_pattern_info(), detect_drift(), and similar audit-mode-only
        # methods work on the model handed back here without the caller
        # needing to know to ask for that separately.
        if not _caller_set_execution_mode:
            refit_params["execution_mode"] = "audit"
        # Keep user-supplied B if present; adaptive_binning ignores it for transaction B.
        best_model = cls(**refit_params).fit(X_train, y_train_arr)

    result = {
        "best_model": best_model,
        "best_params": best_params,
        "best_score": float(best_score),
        "cv_results": rows if return_results else None,
        "cache_fit_seconds_by_G_L_topK": cache_fit_seconds,
        "cache_topK_strategy": "exact_per_G_L_topK_utility_ordered",
        "native_augmented_pair_cache_stats": (
            native_augmented_pair_cache.stats() if native_augmented_pair_cache is not None else None
        ),
        "adaptive_context_seconds": float(adaptive_context_seconds),
        "adaptive_context_used": bool(adaptive_context is not None and "X_pre" in adaptive_context),
        "adaptive_context_misses": (
            int(adaptive_context.get("misses", 0)) if isinstance(adaptive_context, dict) else 0
        ),
        "adaptive_context_hits": (
            max(
                0,
                sum(1 for _g, _l, _k in needed_cache_keys if int(_l) > 1)
                - int(adaptive_context.get("misses", 0)),
            )
            if isinstance(adaptive_context, dict) and "X_pre" in adaptive_context
            else 0
        ),
        "transaction_cache_entries": len(transaction_cache),
        "validation_cache_entries": len(validation_cache) if validation_cache is not None else 0,
        "prep_cache_disabled": bool(disable_prep_cache),
        "validation_cache_disabled": bool(disable_validation_cache),
        "elapsed_seconds": time.perf_counter() - t_start,
        "method": "exact_cached_adaptive_grid",
        "scoring": str(scoring),
    }
    return result


@dataclasses.dataclass
class HUGIMLTuneResult:
    """Result object returned by HUGIMLClassifier.tune().

    Attributes mirror the small subset of GridSearchCV-style fields users need
    for quick HUGIML tuning while keeping the API lightweight.
    """

    best_estimator_: HUGIMLClassifier
    best_params_: dict[str, Any]
    best_score_: float
    results_: Any
    fast_path_used_: bool
    elapsed_seconds_: float
    n_splits_: int
    scoring: str
    cv_splits_: list[tuple[np.ndarray, np.ndarray]]
    shuffle: bool
    random_state: int | None

    # Backward-compatible aliases for dict-style code in notebooks.
    @property
    def best_model(self) -> HUGIMLClassifier:
        return self.best_estimator_

    @property
    def best_params(self) -> dict[str, Any]:
        return self.best_params_

    @property
    def best_score(self) -> float:
        return self.best_score_

    @property
    def cv_results(self) -> Any:
        return self.results_

    @property
    def cv_splits(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.cv_splits_


def _hugiml_params_key(params: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Stable hashable key for parameter dictionaries used in tuning results."""
    return tuple(sorted((str(k), repr(v)) for k, v in dict(params).items()))


def _hugiml_score_model_for_tune(
    model: HUGIMLClassifier,
    X_val: Any,
    y_val: Any,
    scoring: str,
    validation_cache: dict[Any, Any] | None = None,
) -> float:
    """Score one fitted model for HUGIMLClassifier.tune()."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    scoring_norm = str(scoring).lower()
    pred = None
    proba = None
    if (
        validation_cache is not None
        and getattr(model, "max_predict_ms", None) is None
        and not model._is_constant_prior_fallback_active()
    ):
        try:
            X_original = X_val
            X_for_hug = X_val
            if getattr(model, "adaptive_binning", False) and getattr(model, "_bin_edges_", None):
                pre_key = ("prebin", id(getattr(model, "td_", None)))
                X_for_hug = validation_cache.get(pre_key)
                if X_for_hug is None:
                    X_for_hug = model._prebin_for_predict(X_val)
                    validation_cache[pre_key] = X_for_hug
            pat_sig = tuple(
                tuple(int(x) for x in getattr(p, "items", []))
                for p in getattr(model, "patterns_", [])
            )
            z_key = ("hup", id(getattr(model, "td_", None)), pat_sig)
            Z_val = validation_cache.get(z_key)
            if Z_val is None:
                Z_val = model._build_test_hup(X_for_hug)
                validation_cache[z_key] = Z_val
            X_downstream = model._make_downstream_features(X_original, Z_val, fit=False)
            X_downstream = model._apply_strict_topk_budget_transform(X_downstream)
            if scoring_norm in {"roc_auc", "auc"}:
                proba = np.asarray(model.model_.predict_proba(X_downstream))
            else:
                pred = np.asarray(model.model_.predict(X_downstream))
        except Exception:
            # Scoring cache is an optimization only.  Fall back to the public
            # prediction API so correctness follows the existing contract.
            pred = None
            proba = None
    if scoring_norm in {"roc_auc", "auc"}:
        if proba is None:
            proba = model.predict_proba(X_val)
        return _hugiml_auc_score_for_fast_grid(y_val, proba, model.classes_)
    if pred is None:
        pred = model.predict(X_val)
    if scoring_norm == "accuracy":
        return float(accuracy_score(y_val, pred))
    if scoring_norm == "balanced_accuracy":
        return float(balanced_accuracy_score(y_val, pred))
    if scoring_norm in {"f1", "f1_binary"}:
        return float(f1_score(y_val, pred))
    if scoring_norm == "f1_macro":
        return float(f1_score(y_val, pred, average="macro"))
    if scoring_norm == "f1_weighted":
        return float(f1_score(y_val, pred, average="weighted"))
    raise HUGIMLParamError(
        "Unsupported scoring value. Supported: 'roc_auc', 'accuracy', "
        "'balanced_accuracy', 'f1', 'f1_macro', 'f1_weighted'."
    )


def _hugiml_standard_grid_tune_one_split(
    cls,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    candidates: list[dict[str, Any]],
    base_params: dict[str, Any],
    scoring: str,
) -> dict[str, Any]:
    """Ordinary per-candidate grid evaluation for grids not eligible for fast path."""
    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_model: HUGIMLClassifier | None = None
    best_params: dict[str, Any] | None = None
    y_train_arr = cls._safe_cast_y(y_train)
    for candidate_params in candidates:
        t_cand = time.perf_counter()
        params = dict(base_params)
        params.update(candidate_params)
        status = "ok"
        err = None
        score = np.nan
        model = None
        try:
            model = cls(**params).fit(X_train, y_train_arr)
            score = _hugiml_score_model_for_tune(model, X_val, y_val, scoring)
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                best_model = model
                best_params = dict(candidate_params)
        except Exception as exc:
            status = "failed"
            err = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "params": dict(candidate_params),
                "L": candidate_params.get("L", params.get("L")),
                "topK": candidate_params.get("topK", params.get("topK")),
                "feature_mode": candidate_params.get("feature_mode", params.get("feature_mode")),
                "mean_test_score": score,
                "status": status,
                "error": err,
                "elapsed_seconds": time.perf_counter() - t_cand,
            }
        )
    if best_model is None or best_params is None:
        raise HUGIMLValidationError("All tune candidates failed on a validation split.")
    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_score": float(best_score),
        "cv_results": rows,
        "elapsed_seconds": sum(float(r["elapsed_seconds"]) for r in rows),
        "method": "ordinary_grid",
    }


def _hugiml_tune(
    cls,
    X: Any,
    y: Any,
    *,
    cv: int | Any = 5,
    scoring: str = "roc_auc",
    param_grid: dict[str, list] | str | None = None,
    refit: bool = True,
    base_params: dict[str, Any] | None = None,
    random_state: int | None = 42,
    shuffle: bool = True,
    cv_splits: list[tuple[Any, Any]] | None = None,
    use_fast_path: bool = True,
    return_dataframe: bool = True,
) -> HUGIMLTuneResult:
    """Tune HUGIML on full X, y using stratified CV and optional fast-grid caching.

    This is the main public convenience API for quick HUGIML model selection.
    The regular constructor remains a single-configuration estimator; this
    method owns grid search, cross-validation, aggregation, and optional refit.

    Parameters
    ----------
    X, y : array-like or DataFrame/Series
        Full training data.
    cv : int or splitter, default=5
        Number of stratified folds, or any sklearn-compatible splitter with
        split(X, y). Integer cv uses StratifiedKFold.
    scoring : {'roc_auc', 'accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_weighted'}
        Validation metric. 'roc_auc' supports binary and multiclass OVR macro AUC.
    param_grid : dict or None
        A dict (sklearn-style grid), the name of a grid registered in
        hugiml.hyperparameter_configs (for example 'interpretability'),
        or None to use HUGIMLClassifier.default_param_grid().
    refit : bool, default=True
        If True, refit the best configuration on the full X, y with normal fit().
    base_params : dict or None
        Constructor parameters shared by every candidate. ``execution_mode``
        defaults to ``'production'`` for the candidates evaluated during the
        search, since per-candidate line-level allocation diagnostics are
        rarely useful mid-sweep; mining results, scores, and the resulting
        ranking are identical to ``'audit'`` either way. ``result.best_estimator_``
        -- the one model actually returned -- is refit under
        ``execution_mode='audit'`` regardless, so get_pattern_info(),
        detect_drift(), get_drift_psi(), and feature_importances() all work
        on it without missing-artifact warnings, unless ``base_params``
        explicitly names an ``execution_mode``, in which case that value is
        used everywhere (search and final refit alike) and the above
        defaulting is skipped entirely.
    random_state : int or None, default=42
        Random seed for StratifiedKFold when cv is an integer.
    shuffle : bool, default=True
        Whether StratifiedKFold shuffles before splitting.
    cv_splits : list of (train_idx, val_idx) or None, default=None
        Exact fold indices to use. When supplied, cv, shuffle, and random_state
        are ignored for split generation, and the same indices are returned in
        ``result.cv_splits_`` for reuse by other models.
    use_fast_path : bool, default=True
        Use exact cached fast-grid evaluation when the grid qualifies; otherwise
        fall back to ordinary per-candidate evaluation.
    return_dataframe : bool, default=True
        Return ``results_`` as a pandas DataFrame when pandas is available.

    Returns
    -------
    HUGIMLTuneResult
        GridSearchCV-like result object with ``best_estimator_``, ``best_params_``, ``best_score_``, ``results_``,
        ``fast_path_used_``, ``elapsed_seconds_``, and ``n_splits_``.
    """
    from sklearn.model_selection import StratifiedKFold

    t_start = time.perf_counter()
    base_params0 = dict(base_params or {})
    # Recorded before the setdefault below for the same reason as in
    # fast_grid_tune: the final refit further down restores 'audit' by
    # default even though every candidate during the search uses
    # 'production', unless the caller explicitly chose an execution_mode.
    _caller_set_execution_mode = "execution_mode" in base_params0
    base_params0.setdefault("execution_mode", "production")
    candidates = _hugiml_expand_grid_for_fast_tune(param_grid)

    y_arr = cls._safe_cast_y(y)
    n_samples = len(y_arr)
    if cv_splits is not None:
        splits = []
        for split_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
            tr = np.asarray(train_idx, dtype=np.int64)
            va = np.asarray(val_idx, dtype=np.int64)
            if tr.ndim != 1 or va.ndim != 1:
                raise HUGIMLParamError(
                    "Each cv_splits entry must contain 1D train and validation indices."
                )
            if tr.size == 0 or va.size == 0:
                raise HUGIMLParamError(
                    f"cv_splits entry {split_idx} has an empty train or validation index."
                )
            if (
                np.any(tr < 0)
                or np.any(va < 0)
                or np.any(tr >= n_samples)
                or np.any(va >= n_samples)
            ):
                raise HUGIMLParamError(
                    f"cv_splits entry {split_idx} contains indices outside [0, n_samples)."
                )
            if np.intersect1d(tr, va).size > 0:
                raise HUGIMLParamError(
                    f"cv_splits entry {split_idx} has overlapping train and validation indices."
                )
            splits.append((tr.copy(), va.copy()))
    elif isinstance(cv, int):
        if cv < 2:
            raise HUGIMLParamError("cv must be >= 2 when provided as an integer.")
        splitter = StratifiedKFold(
            n_splits=int(cv), shuffle=bool(shuffle), random_state=random_state
        )
        splits = [
            (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
            for tr, va in splitter.split(X, y_arr)
        ]
    else:
        splits = [
            (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
            for tr, va in cv.split(X, y_arr)
        ]
    if not splits:
        raise HUGIMLParamError("cv produced no splits.")

    def _take_rows(obj: Any, idx: np.ndarray) -> Any:
        if hasattr(obj, "iloc"):
            return obj.iloc[idx]
        return np.asarray(obj)[idx]

    fast_path_allowed = False
    if use_fast_path:
        try:
            _hugiml_validate_fast_tune_grid(candidates)
            # The cached fast path is exact only for adaptive-binning grids with
            # no fit-time timeout/degradation. Candidate grids that explicitly
            # set adaptive_binning=False are rejected above; this additional
            # guard covers the common case where adaptive_binning or
            # max_fit_seconds/max_mining_seconds may be supplied only through base_params.
            if (
                bool(base_params0.get("adaptive_binning", True))
                and base_params0.get("max_fit_seconds", None) is None
                and base_params0.get("max_mining_seconds", None) is None
            ):
                fast_path_allowed = True
        except Exception:
            fast_path_allowed = False

    fold_rows: list[dict[str, Any]] = []
    fold_methods: list[str] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train = _take_rows(X, train_idx)
        X_val = _take_rows(X, val_idx)
        y_train = y_arr[train_idx]
        y_val = y_arr[val_idx]
        if fast_path_allowed:
            try:
                split_result = cls.fast_grid_tune(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    param_grid=param_grid,
                    base_params=base_params0,
                    scoring=scoring,
                    refit_full=False,
                    return_results=True,
                )
            except Exception:
                # Preserve correctness over speed: an unexpected cached-path
                # failure for one fold should fall back to the ordinary
                # per-candidate evaluation rather than aborting tuning.
                fast_path_allowed = False
                split_result = _hugiml_standard_grid_tune_one_split(
                    cls,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    candidates,
                    base_params0,
                    scoring,
                )
        else:
            split_result = _hugiml_standard_grid_tune_one_split(
                cls,
                X_train,
                y_train,
                X_val,
                y_val,
                candidates,
                base_params0,
                scoring,
            )
        fold_methods.append(str(split_result.get("method", "unknown")))
        for row in split_result.get("cv_results") or []:
            params = dict(row.get("params", {}))
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "params_key": _hugiml_params_key(params),
                    "params": params,
                    "L": row.get("L", params.get("L")),
                    "topK": row.get("topK", params.get("topK")),
                    "feature_mode": row.get("feature_mode", params.get("feature_mode")),
                    "split_test_score": row.get("mean_test_score", np.nan),
                    "status": row.get("status", "ok"),
                    "error": row.get("error"),
                    "elapsed_seconds": row.get("elapsed_seconds", np.nan),
                }
            )

    if not fold_rows:
        raise HUGIMLValidationError("No tuning results were produced.")

    grouped: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
    for row in fold_rows:
        grouped.setdefault(row["params_key"], []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows_for_key in grouped.items():
        scores = np.asarray([float(r["split_test_score"]) for r in rows_for_key], dtype=float)
        finite = scores[np.isfinite(scores)]
        first_params = dict(rows_for_key[0]["params"])
        summary_rows.append(
            {
                "params": first_params,
                "L": first_params.get("L"),
                "topK": first_params.get("topK"),
                "feature_mode": first_params.get("feature_mode"),
                "mean_test_score": float(np.mean(finite)) if finite.size else np.nan,
                "std_test_score": float(np.std(finite, ddof=0)) if finite.size else np.nan,
                "n_successful_splits": int(finite.size),
                "n_splits": int(len(splits)),
                "mean_elapsed_seconds": float(
                    np.nanmean([r["elapsed_seconds"] for r in rows_for_key])
                ),
                "status": "ok" if finite.size == len(splits) else "partial_or_failed",
            }
        )
    summary_rows.sort(
        key=lambda r: (
            -float(r["mean_test_score"]) if np.isfinite(r["mean_test_score"]) else np.inf,
            repr(r["params"]),
        )
    )
    if not summary_rows or not np.isfinite(summary_rows[0]["mean_test_score"]):
        raise HUGIMLValidationError("All tune candidates failed across CV splits.")
    for rank, row in enumerate(summary_rows, start=1):
        row["rank_test_score"] = rank

    best_params = dict(base_params0)
    best_params.update(dict(summary_rows[0]["params"]))
    best_score = float(summary_rows[0]["mean_test_score"])
    # The search above evaluates every candidate under 'production' by
    # default for speed; best_estimator_ is the model the caller actually
    # keeps and is the natural target for get_pattern_info(), detect_drift(),
    # and similar audit-mode-only methods afterward, so unless the caller's
    # own base_params explicitly named an execution_mode, it's refit under
    # 'audit' here rather than inheriting the search's speed-oriented
    # default.
    if not _caller_set_execution_mode:
        best_params["execution_mode"] = "audit"
    if refit:
        best_estimator = cls(**best_params).fit(X, y_arr)
    else:
        # Return a fitted estimator from the first fold for convenience.  It is
        # valid for immediate inspection/prediction on that fold's fitted state,
        # but refit=True is recommended for production use.
        train_idx, val_idx = splits[0]
        best_estimator = cls(**best_params).fit(_take_rows(X, train_idx), y_arr[train_idx])

    if return_dataframe:
        try:
            results_obj = pd.DataFrame(summary_rows)
        except Exception:
            results_obj = summary_rows
    else:
        results_obj = summary_rows

    return HUGIMLTuneResult(
        best_estimator_=best_estimator,
        best_params_=best_params,
        best_score_=best_score,
        results_=results_obj,
        fast_path_used_=bool(
            fast_path_allowed and all(m == "exact_cached_adaptive_grid" for m in fold_methods)
        ),
        elapsed_seconds_=time.perf_counter() - t_start,
        n_splits_=int(len(splits)),
        scoring=str(scoring),
        cv_splits_=[(tr.copy(), va.copy()) for tr, va in splits],
        shuffle=bool(shuffle),
        random_state=random_state,
    )
