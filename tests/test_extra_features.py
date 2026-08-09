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

Run directly:

    python3 test_extra_features.py

or under pytest:

    pytest test_extra_features.py -v

Every check here uses synthetic, generated-in-place data rather than files
specific to any one machine, so this runs the same way against any source
checkout. Some sections are optional and skip themselves
with a printed reason rather than failing outright:

- The native-function section needs the compiled extension
  (``_hugiml_core``) importable; if it isn't, that section is skipped.
- The dashboard sections need ``streamlit`` installed; if it isn't, they're
  skipped.

Sections
--------
1. Native categorical-extraction correctness and consistency
2. Zero-variance column exclusion
3. Hyperparameter grid configuration
4. Tuning execution_mode threading
5. Dashboard governance cold-start message
6. Dashboard Workbench tuning-grid selector
"""

from __future__ import annotations

import os
import sys
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from hugiml import HUGIMLClassifier, check_native
from hugiml import classifier as clf_mod
from hugiml.benchmarks import runner as benchmark_runner
from hugiml.classifier import (
    NativeAugmentedPairTransformBlock,
    _best_ig_score,
    _codes_from_edges,
    _continuous_to_quantile_codes,
    _dense_full_csr,
    _edge_information_gain,
    _entropy_from_counts,
    _get_peak_rss_kb,
    _information_gain_from_codes,
    _is_binary_feature_series,
    _joint_information_gain_from_binned_columns,
)
from hugiml.hyperparameter_configs import (
    BASELINE_MODEL_GRIDS,
    BUDGETED_BASELINE_MODEL_GRIDS,
    DEFAULT_HUGIML_GRID_NAME,
    get_baseline_grid,
    get_budgeted_baseline_grid,
    get_hugiml_grid,
    list_hugiml_grids,
)

pd.set_option("future.infer_string", False)

PASS = []
FAIL = []


def _report(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    (PASS if ok else FAIL).append(name)


def _skip(name: str, reason: str) -> None:
    print(f"[SKIP] {name} -- {reason}")


# =============================================================================
# Section 1: native categorical-extraction correctness and consistency
# =============================================================================


def _make_native_cases() -> dict[str, dict[str, Any]]:
    """Synthetic cases spanning the shapes the native changes touch."""
    cases: dict[str, dict[str, Any]] = {}

    def build(n, specs, seed):
        r = np.random.default_rng(seed)
        p = len(specs)
        X_num = np.zeros((n, p), dtype=np.float64)
        is_cat = np.zeros(p, dtype=np.uint8)
        is_int = np.zeros(p, dtype=np.uint8)
        is_precoded = np.zeros(p, dtype=np.uint8)
        X_cat_raw = [None] * p
        col_names = []
        for j, (kind, params) in enumerate(specs):
            col_names.append(f"c{j}_{kind}")
            if kind == "num":
                X_num[:, j] = r.normal(size=n)
            elif kind == "num_missing":
                vals = r.normal(size=n)
                miss = r.choice(n, size=max(1, n // 10), replace=False)
                vals[miss] = np.nan
                X_num[:, j] = vals
            elif kind == "precoded":
                nb = params.get("nb", 5)
                X_num[:, j] = r.integers(0, nb, size=n).astype(np.float64)
                is_precoded[j] = 1
            elif kind in ("cat", "cat_missing"):
                levels = params.get("levels", 8)
                cats = [f"v{k}" for k in range(levels)]
                vals = r.choice(cats, size=n).astype(object)
                if kind == "cat_missing":
                    miss = r.choice(n, size=max(1, n // 10), replace=False)
                    vals[miss] = None
                X_cat_raw[j] = vals
                is_cat[j] = 1
            else:
                raise ValueError(kind)
        return dict(
            X_num=X_num,
            col_names=col_names,
            is_cat=is_cat,
            is_int=is_int,
            is_precoded=is_precoded,
            X_cat_raw=X_cat_raw,
        )

    c = build(1500, [("num", {})] * 5, seed=1)
    c["y"] = np.random.default_rng(1).integers(0, 2, size=1500).astype(np.int64)
    cases["pure_numeric"] = c

    c = build(1500, [("num_missing", {})] * 5, seed=2)
    c["y"] = np.random.default_rng(2).integers(0, 2, size=1500).astype(np.int64)
    cases["pure_numeric_missing"] = c

    c = build(1500, [("cat", {"levels": 8})] * 5, seed=3)
    c["y"] = np.random.default_rng(3).integers(0, 2, size=1500).astype(np.int64)
    cases["pure_categorical"] = c

    c = build(1500, [("cat_missing", {"levels": 8})] * 5, seed=4)
    c["y"] = np.random.default_rng(4).integers(0, 2, size=1500).astype(np.int64)
    cases["pure_categorical_missing"] = c

    specs = [
        ("num", {}),
        ("cat", {"levels": 6}),
        ("cat_missing", {"levels": 6}),
        ("precoded", {"nb": 4}),
    ]
    c = build(2000, specs, seed=5)
    c["y"] = np.random.default_rng(5).integers(0, 2, size=2000).astype(np.int64)
    cases["mixed_with_missing"] = c

    c = build(2000, [("cat", {"levels": 300}), ("num", {})], seed=6)
    c["y"] = np.random.default_rng(6).integers(0, 2, size=2000).astype(np.int64)
    cases["high_cardinality"] = c

    specs = [("cat_missing", {"levels": 5}), ("num", {})]
    c = build(300, specs, seed=7)
    c["X_cat_raw"][0] = np.array([None] * 300, dtype=object)
    c["y"] = np.random.default_rng(7).integers(0, 2, size=300).astype(np.int64)
    cases["categorical_all_missing"] = c

    c = build(300, [("cat", {"levels": 1}), ("num", {})], seed=8)
    c["y"] = np.random.default_rng(8).integers(0, 2, size=300).astype(np.int64)
    cases["categorical_single_level"] = c

    c = build(2, [("cat", {"levels": 3}), ("num", {})], seed=9)
    c["y"] = np.array([0, 1], dtype=np.int64)
    cases["tiny_two_rows"] = c

    return cases


def _td_signature(td) -> dict:
    return dict(
        n_transactions=len(td.transactions),
        item_twu=[round(float(x), 6) for x in td.item_twu],
        item_map=dict(td.item_map),
        bn2id=dict(td.bn2id),
        nb_col=list(td.nb_col),
        is_cat_v=list(td.is_cat_v),
        cat_categories=[list(c) for c in td.cat_categories],
    )


def run_native_section() -> None:
    print("\n--- Section 1: native categorical extraction ---")
    try:
        import _hugiml_core as core
    except ImportError:
        _skip("native section", "_hugiml_core is not importable")
        return

    baseline_dir = os.environ.get("HUGIML_BASELINE_NATIVE_DIR")
    baseline_core = None
    if baseline_dir:
        sys.path.insert(0, baseline_dir)
        try:
            import importlib

            baseline_core = importlib.import_module("_hugiml_core")
        except ImportError:
            print(
                "  (HUGIML_BASELINE_NATIVE_DIR set but no _hugiml_core found there; "
                "skipping the before/after timing comparison)"
            )
        finally:
            sys.path.remove(baseline_dir)

    cases = _make_native_cases()
    for name, case in cases.items():
        try:
            td = core.prepare_transactions(
                case["X_num"],
                case["y"],
                2,
                case["col_names"],
                case["is_cat"],
                case["is_int"],
                case["X_cat_raw"],
                case["is_precoded"],
            )
            _report(f"prepare_transactions: {name}", True, f"n_items={len(td.item_twu)}")
        except Exception as exc:
            _report(f"prepare_transactions: {name}", False, f"{type(exc).__name__}: {exc}")
            continue

        try:
            n_cls = len(set(case["y"].tolist()))
            y_codes = pd.factorize(case["y"], sort=True)[0].astype("int64")
            K = max(1, min(50, len(td.item_twu)))
            patterns_generic = list(core.mine_patterns(td, y_codes, n_cls, K, 2, 0.0001, 0.0))
            patterns_l2 = list(core.mine_patterns_l2(td, y_codes, n_cls, K, 0.0001, 0.0))
            sig_generic = sorted(
                (round(p.utility, 6), tuple(sorted(p.items))) for p in patterns_generic
            )
            sig_l2 = sorted((round(p.utility, 6), tuple(sorted(p.items))) for p in patterns_l2)
            _report(
                f"mine_patterns(L=2) == mine_patterns_l2: {name}",
                sig_generic == sig_l2,
                f"{len(patterns_generic)} patterns",
            )
        except Exception as exc:
            _report(f"mining consistency: {name}", False, f"{type(exc).__name__}: {exc}")

        if baseline_core is not None:
            try:
                td_base = baseline_core.prepare_transactions(
                    case["X_num"],
                    case["y"],
                    2,
                    case["col_names"],
                    case["is_cat"],
                    case["is_int"],
                    case["X_cat_raw"],
                    case["is_precoded"],
                )
                match = _td_signature(td) == _td_signature(td_base)
                _report(f"matches baseline build: {name}", match)
            except Exception as exc:
                _report(f"matches baseline build: {name}", False, f"{type(exc).__name__}: {exc}")


# =============================================================================
# Section 2: zero-variance column exclusion
# =============================================================================


def run_zero_variance_section() -> None:
    print("\n--- Section 2: zero-variance column exclusion ---")
    try:
        from hugiml import HUGIMLClassifier
    except ImportError as exc:
        _skip("zero-variance section", f"hugiml not importable ({exc})")
        return

    rng = np.random.default_rng(0)
    n = 1200
    X = pd.DataFrame(
        {
            "real_cat": rng.choice([f"v{i}" for i in range(8)], size=n),
            "real_num": rng.normal(size=n),
            "constant_cat": ["ALWAYS_SAME"] * n,
            "constant_num": np.full(n, 7.0),
            "all_missing_cat": pd.array([None] * n, dtype=object),
        }
    )
    y = rng.integers(0, 2, size=n).astype(np.int64)
    X_test = X.iloc[:200].copy()

    params = dict(
        B=-1,
        adaptive_binning=True,
        L=2,
        topK=50,
        feature_mode="original_plus_patterns",
        G=0.01,
        execution_mode="production",
    )

    clf = HUGIMLClassifier(**params)
    clf.fit(X, y)
    detected = set(getattr(clf, "_zero_variance_cols_", []))
    expected = {"constant_cat", "constant_num", "all_missing_cat"}
    _report(
        "zero-variance columns detected correctly",
        detected == expected,
        f"detected={sorted(detected)}",
    )
    proba_with_exclusion = clf.predict_proba(X_test)

    # Toggle the exclusion off within the same process (no second codebase
    # needed) by making the identification step report nothing, then compare.
    clf_noexcl = HUGIMLClassifier(**params)
    clf_noexcl._identify_zero_variance_columns = lambda X_train: []
    clf_noexcl.fit(X, y)
    proba_without_exclusion = clf_noexcl.predict_proba(X_test)

    identical = np.allclose(proba_with_exclusion, proba_without_exclusion, atol=1e-9)
    _report("predictions identical with vs without exclusion", identical)

    same_pattern_count = clf.fit_metadata_.n_patterns == clf_noexcl.fit_metadata_.n_patterns
    _report("mined pattern count unaffected by exclusion", same_pattern_count)

    # Predict-time consistency: the model should ignore these columns even
    # if new data gives them non-constant values.
    X_test_drift = X_test.copy()
    X_test_drift["constant_cat"] = rng.choice(["A", "B", "C"], size=len(X_test_drift))
    X_test_drift["constant_num"] = rng.normal(size=len(X_test_drift))
    try:
        proba_drift = clf.predict_proba(X_test_drift)
        ran_ok = proba_drift.shape == proba_with_exclusion.shape
        _report("predict() tolerates drifted values in excluded columns", ran_ok)
    except Exception as exc:
        _report(
            "predict() tolerates drifted values in excluded columns",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    # Serialization round trip: _zero_variance_cols_ is fitted state derived
    # from training data, not re-derivable from a loaded model alone, so it
    # needs to be saved and restored explicitly rather than left to whatever
    # a generic save mechanism happens to carry over.
    try:
        import tempfile

        from hugiml.serialization import load_model, save_model

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.hugiml")
            save_model(clf, path)
            clf_loaded = load_model(path)
            _report(
                "zero-variance column list survives save/load",
                set(getattr(clf_loaded, "_zero_variance_cols_", [])) == detected,
            )
            proba_loaded = clf_loaded.predict_proba(X_test)
            _report(
                "predictions identical after save/load round trip",
                np.allclose(proba_with_exclusion, proba_loaded, atol=1e-9),
            )
    except ImportError as exc:
        _skip("serialization round trip", f"not importable ({exc})")
    except Exception as exc:
        _report("serialization round trip", False, f"{type(exc).__name__}: {exc}")


# =============================================================================
# Section 3: hyperparameter grid configuration
# =============================================================================


def run_grid_config_section() -> None:
    print("\n--- Section 3: hyperparameter grid configuration ---")
    try:
        from hugiml import HUGIMLClassifier
        from hugiml.hyperparameter_configs import (
            DEFAULT_HUGIML_GRID_NAME,
            get_baseline_grid,
            get_hugiml_grid,
            list_hugiml_grids,
        )
    except ImportError as exc:
        _skip("grid configuration section", f"not importable ({exc})")
        return

    expected_performance = {
        "B": [-1],
        "adaptive_binning": [True],
        "L": [1, 2],
        "topK": [50, 100],
        "feature_mode": ["original_plus_patterns"],
        "G": [0.01, 0.001],
        "convert_binary_to_categorical": [False],
    }
    _report(
        "performance grid matches the specified values",
        get_hugiml_grid("performance") == expected_performance,
    )
    _report("default grid name is 'performance_ho'", DEFAULT_HUGIML_GRID_NAME == "performance_ho")
    _report(
        "default_param_grid() with no name equals the performance_ho grid",
        HUGIMLClassifier.default_param_grid() == get_hugiml_grid("performance_ho"),
    )

    interp = get_hugiml_grid("interpretability")
    _report(
        "interpretability grid uses patterns_only",
        interp.get("feature_mode") == ["patterns_only"],
    )
    _report(
        "interpretability grid sets interaction_relaxed_mining",
        interp.get("interaction_relaxed_mining") == [True],
    )
    _report(
        "interpretability grid disables augmented_pair_transforms "
        "(required: it's mutually exclusive with interaction_relaxed_mining at L>=2)",
        interp.get("augmented_pair_transforms") == [False],
    )

    try:
        get_hugiml_grid("not_a_real_grid")
        _report("unknown grid name raises KeyError", False, "no exception raised")
    except KeyError:
        _report("unknown grid name raises KeyError", True)

    _report(
        "baseline grid lookup returns None for an unregistered model",
        get_baseline_grid("SomeModelNotInTheRegistry") is None,
    )
    _report(
        "baseline grid lookup returns a grid for a registered model",
        isinstance(get_baseline_grid("XGBoost"), dict),
    )
    _report(
        "list_hugiml_grids includes all four named grids",
        set(list_hugiml_grids())
        == {"performance", "interpretability", "performance_ho", "interpretability_ho"},
    )

    # Validate every candidate in the interpretability grid actually
    # constructs without HUGIMLClassifier rejecting the combination.
    try:
        from itertools import product

        keys = list(interp.keys())
        bad = []
        for values in product(*[interp[k] for k in keys]):
            params = dict(zip(keys, values))
            try:
                HUGIMLClassifier(**params)
            except Exception as exc:
                bad.append((params, str(exc)))
        _report(
            "every interpretability-grid candidate constructs without error",
            not bad,
            f"{len(bad)} failing combinations" if bad else "",
        )
    except Exception as exc:
        _report("every interpretability-grid candidate constructs without error", False, str(exc))


# =============================================================================
# Section 4: tuning execution_mode threading
# =============================================================================


def run_execution_mode_section() -> None:
    print("\n--- Section 4: tuning execution_mode threading ---")
    try:
        from hugiml import HUGIMLClassifierNative
    except ImportError as exc:
        _skip("execution_mode section", f"hugiml not importable ({exc})")
        return

    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(
        {
            "cat1": rng.choice([f"v{i}" for i in range(5)], size=n),
            "num1": rng.normal(size=n),
        }
    )
    y = rng.integers(0, 2, size=n).astype(np.int64)

    try:
        result = HUGIMLClassifierNative.tune(
            X, y, cv=2, scoring="roc_auc", param_grid="performance"
        )
        _report(
            "tune() default: best_estimator_ is execution_mode='audit'",
            result.best_estimator_.execution_mode == "audit",
            f"got {result.best_estimator_.execution_mode!r}",
        )
        try:
            result.best_estimator_.get_pattern_info()
            _report("tune() default: get_pattern_info() works on best_estimator_", True)
        except Exception as exc:
            _report(
                "tune() default: get_pattern_info() works on best_estimator_", False, str(exc)[:100]
            )
        _report(
            "tune() with param_grid='performance' (string) runs the fast path",
            bool(result.fast_path_used_),
        )
    except Exception as exc:
        _report("tune() with default execution_mode", False, f"{type(exc).__name__}: {exc}")

    try:
        result_prod = HUGIMLClassifierNative.tune(
            X, y, cv=2, scoring="roc_auc", base_params={"execution_mode": "production"}
        )
        _report(
            "explicit execution_mode='production' is honored end-to-end on best_estimator_",
            result_prod.best_estimator_.execution_mode == "production",
        )
    except Exception as exc:
        _report(
            "explicit execution_mode='production' is honored end-to-end",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    try:
        result_audit = HUGIMLClassifierNative.tune(
            X, y, cv=2, scoring="roc_auc", base_params={"execution_mode": "audit"}
        )
        _report(
            "explicit execution_mode='audit' is respected, not overridden",
            result_audit.best_params_.get("execution_mode") == "audit",
        )
    except Exception as exc:
        _report(
            "explicit execution_mode='audit' is respected", False, f"{type(exc).__name__}: {exc}"
        )

    try:
        result_interp = HUGIMLClassifierNative.tune(
            X, y, cv=2, scoring="roc_auc", param_grid="interpretability"
        )
        _report(
            "tune() with param_grid='interpretability' (string) succeeds",
            result_interp.best_params_.get("feature_mode") == "patterns_only",
        )
    except Exception as exc:
        _report("tune() with param_grid='interpretability'", False, f"{type(exc).__name__}: {exc}")

    try:
        fgt_default = HUGIMLClassifierNative.fast_grid_tune(
            X,
            y,
            X.iloc[:100],
            y[:100],
            param_grid="performance",
        )
        _report(
            "fast_grid_tune(refit_full=False, default) candidate stays execution_mode='production'",
            fgt_default["best_model"].execution_mode == "production",
        )
        fgt_refit = HUGIMLClassifierNative.fast_grid_tune(
            X,
            y,
            X.iloc[:100],
            y[:100],
            param_grid="performance",
            refit_full=True,
        )
        _report(
            "fast_grid_tune(refit_full=True) default: refit model is execution_mode='audit'",
            fgt_refit["best_model"].execution_mode == "audit",
        )
        try:
            fgt_refit["best_model"].get_pattern_info()
            _report("fast_grid_tune(refit_full=True): get_pattern_info() works", True)
        except Exception as exc:
            _report(
                "fast_grid_tune(refit_full=True): get_pattern_info() works", False, str(exc)[:100]
            )
    except Exception as exc:
        _report("fast_grid_tune refit_full behavior", False, f"{type(exc).__name__}: {exc}")


# =============================================================================
# Section 5: dashboard governance cold-start message
# =============================================================================

EXPECTED_COLD_START_MESSAGE = (
    "No promoted HUGIML run yet. Go to Workbench \u2192 Results \u2192 "
    "select a HUGIML run \u2192 Promote to Governance."
)


def _find_dashboard_app_path() -> str | None:
    try:
        import hugiml.dashboard.app as app_module

        return app_module.__file__
    except ImportError:
        return None


def run_governance_section() -> None:
    print("\n--- Section 5: dashboard governance cold-start message ---")
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        _skip("governance section", "streamlit is not installed")
        return

    app_path = _find_dashboard_app_path()
    if app_path is None:
        _skip("governance section", "hugiml.dashboard.app is not importable")
        return

    try:
        at = AppTest.from_file(app_path, default_timeout=60)
        at.session_state["hugiml_nav_section"] = "Governance"
        at.run()
        infos = [b.value for b in at.info]
        _report("cold Governance page has no exception", not at.exception)
        _report(
            "cold Governance page shows the expected message",
            EXPECTED_COLD_START_MESSAGE in infos,
            "" if EXPECTED_COLD_START_MESSAGE in infos else f"got: {infos}",
        )
    except Exception as exc:
        _report("cold Governance page renders", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    try:
        at2 = AppTest.from_file(app_path, default_timeout=60)
        at2.session_state["hugiml_nav_section"] = "Workbench"
        at2.run()
        infos2 = [b.value for b in at2.info]
        _report(
            "Workbench workspace does not show the governance message",
            EXPECTED_COLD_START_MESSAGE not in infos2,
        )
    except Exception as exc:
        _report("Workbench workspace renders", False, f"{type(exc).__name__}: {exc}")

    try:
        at3 = AppTest.from_file(app_path, default_timeout=60)
        at3.session_state["hugiml_nav_section"] = "Governance"
        at3.session_state["hugiml_promoted_governance_ctx"] = {"fake": "context"}
        at3.run()
        infos3 = [b.value for b in at3.info]
        _report(
            "Governance with a promoted context does not show the cold-start message",
            EXPECTED_COLD_START_MESSAGE not in infos3,
        )
    except Exception as exc:
        _report("Governance-with-promoted-context check", False, f"{type(exc).__name__}: {exc}")


# =============================================================================
# Section 6: dashboard Workbench tuning-grid selector
# =============================================================================


def run_workbench_grid_selector_section() -> None:
    print("\n--- Section 6: dashboard Workbench tuning-grid selector ---")
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        _skip("workbench grid selector section", "streamlit is not installed")
        return

    app_path = _find_dashboard_app_path()
    if app_path is None:
        _skip("workbench grid selector section", "hugiml.dashboard.app is not importable")
        return

    try:
        at = AppTest.from_file(app_path, default_timeout=120)
        at.session_state["hugiml_nav_section"] = "Workbench"
        at.run()

        hugiml_radios = [r for r in at.radio if r.key == "wb_hugiml_mode"]
        if not hugiml_radios:
            _report("HUGIML configuration-mode radio is present", False)
            return
        hugiml_radios[0].set_value("Guided").run()
        _report("selecting Guided mode does not raise", not at.exception)

        selectors = [sb for sb in at.selectbox if sb.key == "wb_hugiml_grid_name"]
        _report("tuning-grid selector is present in Guided mode", bool(selectors))
        if selectors:
            _report(
                "tuning-grid selector defaults to 'performance_ho'",
                selectors[0].value == "performance_ho",
            )
            _report(
                "tuning-grid selector offers all named grids",
                set(selectors[0].options)
                >= {
                    "performance_ho",
                    "performance",
                    "interpretability_ho",
                    "interpretability",
                },
            )
            selectors[0].set_value("interpretability").run()
            _report("switching to the interpretability grid does not raise", not at.exception)
    except Exception as exc:
        _report("workbench grid selector flow", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


# =============================================================================


def main() -> int:
    run_native_section()
    run_zero_variance_section()
    run_grid_config_section()
    run_execution_mode_section()
    run_governance_section()
    run_workbench_grid_selector_section()

    print(f"\n{'=' * 60}")
    print(f"PASS: {len(PASS)}   FAIL: {len(FAIL)}")
    if FAIL:
        print("Failing checks:")
        for name in FAIL:
            print(f"  - {name}")
    print("=" * 60)
    return 1 if FAIL else 0


# =============================================================================
# Consolidated unit coverage for shared grid and helper behavior
# =============================================================================


def test_hugiml_named_grids_have_expected_values_and_are_copy_safe() -> None:
    performance = get_hugiml_grid("performance")
    assert DEFAULT_HUGIML_GRID_NAME == "performance_ho"
    linear_estimators = performance.pop("base_estimator")
    assert performance == {
        "B": [-1],
        "adaptive_binning": [True],
        "L": [1, 2],
        "topK": [50, 100],
        "feature_mode": ["original_plus_patterns"],
        "G": [0.01, 0.001],
        "convert_binary_to_categorical": [False],
    }
    assert len(linear_estimators) == 1
    linear = linear_estimators[0]
    assert isinstance(linear, LogisticRegression)
    assert linear.solver == "liblinear"
    assert linear.max_iter == 300
    assert linear.random_state == 0

    performance["topK"].append(200)
    assert get_hugiml_grid("performance")["topK"] == [50, 100]

    interpretability = get_hugiml_grid("interpretability")
    assert interpretability["feature_mode"] == ["patterns_only"]
    assert interpretability["interaction_relaxed_mining"] == [True]
    assert interpretability["convert_binary_to_categorical"] == [True]
    assert interpretability["augmented_pair_transforms"] == [False]
    assert set(list_hugiml_grids()) == {
        "performance_ho",
        "performance",
        "interpretability_ho",
        "interpretability",
    }


def test_classifier_default_param_grid_accepts_named_grid() -> None:
    default_grid = HUGIMLClassifier.default_param_grid()
    performance_ho = get_hugiml_grid("performance_ho")
    assert default_grid.keys() == performance_ho.keys()
    for key in default_grid:
        if key == "base_estimator":
            # Each accessor deliberately deep-copies live estimator objects,
            # so identity/value equality is not meaningful here.
            assert len(default_grid[key]) == len(performance_ho[key]) == 2
            assert default_grid[key][0] is None
            assert performance_ho[key][0] is None
            assert type(default_grid[key][1]) is type(performance_ho[key][1])
        else:
            assert default_grid[key] == performance_ho[key]
    for name in ("performance", "interpretability"):
        classifier_grid = HUGIMLClassifier.default_param_grid(name)
        named_grid = get_hugiml_grid(name)
        assert classifier_grid.keys() == named_grid.keys()
        for key in classifier_grid:
            if key == "base_estimator":
                assert [repr(value) for value in classifier_grid[key]] == [
                    repr(value) for value in named_grid[key]
                ]
            else:
                assert classifier_grid[key] == named_grid[key]


def test_fast_tune_grid_expansion_accepts_string_grid() -> None:
    from hugiml.classifier import _hugiml_expand_grid_for_fast_tune

    rows = _hugiml_expand_grid_for_fast_tune("interpretability")
    assert rows
    assert {row["feature_mode"] for row in rows} == {"patterns_only"}
    assert all(row["augmented_pair_transforms"] is False for row in rows)
    assert all(row["convert_binary_to_categorical"] is True for row in rows)


def test_baseline_grids_are_centralized_and_copy_safe() -> None:
    from sklearn.model_selection import ParameterGrid

    assert benchmark_runner.TUNING_GRIDS is BASELINE_MODEL_GRIDS
    rf_grid = get_baseline_grid("RandomForest")
    assert rf_grid is not None
    rf_grid["n_estimators"].append(800)
    assert get_baseline_grid("RandomForest")["n_estimators"] == [200, 400]
    assert get_baseline_grid("UnregisteredModel") is None

    for model_name in ["XGBoost", "LightGBM", "RandomForest"]:
        assert len(list(ParameterGrid(get_baseline_grid(model_name)))) == 16
        budgeted = get_budgeted_baseline_grid(model_name)
        assert budgeted is not None
        budgeted_candidates = list(ParameterGrid(budgeted))
        assert len(budgeted_candidates) == 16
        for candidate in budgeted_candidates:
            if model_name == "XGBoost":
                ceiling = candidate["n_estimators"] * (2 ** candidate["max_depth"])
            elif model_name == "LightGBM":
                ceiling = candidate["n_estimators"] * candidate["num_leaves"]
            else:
                ceiling = candidate["n_estimators"] * candidate["max_leaf_nodes"]
            assert ceiling <= 200

    assert len(list(ParameterGrid(get_baseline_grid("EBM")))) == 8
    assert len(list(ParameterGrid(get_baseline_grid("RuleFit")))) == 8
    assert get_budgeted_baseline_grid("UnregisteredModel") is None

    budgeted_rf = get_budgeted_baseline_grid("RandomForest")
    budgeted_rf["n_estimators"].append(100)
    assert BUDGETED_BASELINE_MODEL_GRIDS["RandomForest"]["n_estimators"] == [25, 50]


def test_baseline_ohe_preprocessor_handles_unknown_categories_and_missing_columns() -> None:
    prep = benchmark_runner._BaselineOHEPreprocessor()
    train = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0],
            "cat": pd.Series(["a", "b", "a", None], dtype="string"),
            "flag": [True, False, True, False],
        }
    )
    transformed_train = prep.fit_transform(train)
    assert transformed_train.shape[0] == len(train)
    assert not transformed_train.isna().any().any()

    test = pd.DataFrame({"num": [np.nan, 5.0], "cat": ["unseen", "a"]})
    transformed_test = prep.transform(test)
    assert list(transformed_test.columns) == list(transformed_train.columns)
    assert transformed_test.shape[0] == 2
    assert not transformed_test.isna().any().any()


def test_zero_variance_detection_handles_dataframe_and_object_arrays() -> None:
    clf = HUGIMLClassifier()
    frame = pd.DataFrame(
        {
            "constant": ["same", "same", None],
            "all_missing": [pd.NA, pd.NA, pd.NA],
            "varying": [1, 2, 1],
        }
    )
    clf.feature_names_in_ = list(frame.columns)
    assert set(clf._identify_zero_variance_columns(frame)) == {"constant", "all_missing"}

    arr = np.array([["same", None, 1], ["same", None, 2], ["same", None, 1]], dtype=object)
    clf.feature_names_in_ = ["constant", "all_missing", "varying"]
    assert set(clf._identify_zero_variance_columns(arr)) == {"constant", "all_missing"}


def test_low_level_scoring_helpers_cover_boundary_paths() -> None:
    assert _best_ig_score({"a": 0.1, "b": "bad", "c": np.inf}) == 0.1
    assert _best_ig_score("not-numeric") == 0.0
    assert _best_ig_score(np.nan) == 0.0

    assert _entropy_from_counts(np.array([0, 0])) == 0.0
    assert _information_gain_from_codes(np.array([-1, -1]), np.array([0, 1]), 2) == 0.0
    assert _information_gain_from_codes(np.array([1, 1, 1]), np.array([0, 0, 0]), 2) == 0.0

    dense = _dense_full_csr(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert dense.shape == (2, 2)
    assert _dense_full_csr(np.zeros((3, 0))).shape == (3, 0)

    all_missing_codes = _continuous_to_quantile_codes(np.array([np.nan, np.inf]))
    assert all_missing_codes.tolist() == [-1, -1]
    assert _continuous_to_quantile_codes(np.array([1.0, 1.0, 1.0]), max_bins=4).tolist() == [
        0,
        0,
        0,
    ]
    many_codes = _continuous_to_quantile_codes(np.arange(20, dtype=float), max_bins=4)
    assert set(many_codes.tolist()) == {0, 1, 2, 3}

    assert _codes_from_edges(np.array([1.0, np.nan]), np.array([0.0])).tolist() == [-1, -1]
    assert _codes_from_edges(np.array([1.0, 2.0]), np.array([0.0, 3.0])).tolist() == [0, 0]


def test_edge_and_joint_information_gain_paths() -> None:
    y = np.array([0, 1] * 8)
    short_score, short_edges, short_codes = _edge_information_gain(
        np.array([1.0, np.nan]), y[:2], 2, 3
    )
    assert short_score == 0.0
    assert short_edges.size >= 2
    assert short_codes.shape == (2,)

    values = np.r_[np.zeros(8), np.ones(8)]
    y2 = np.r_[np.zeros(8, dtype=int), np.ones(8, dtype=int)]
    score, edges, codes = _edge_information_gain(values, y2, 2, 4)
    assert score > 0.0
    assert edges.size >= 2
    assert codes.min() >= 0

    joint, left_score, right_score = _joint_information_gain_from_binned_columns(
        values, values[::-1], y2, 2, 4, 4
    )
    assert joint >= max(left_score, right_score)
    assert (
        _joint_information_gain_from_binned_columns(values[:4], values[:4], y2[:4], 2, 4, 4)[0]
        == 0.0
    )


def test_named_grid_error_and_package_helpers() -> None:
    assert "performance" in list_hugiml_grids()
    assert get_baseline_grid("NoGridModel") is None
    with pytest.raises(KeyError, match="Unknown HUGIML grid"):
        get_hugiml_grid("not-present")
    assert isinstance(check_native(), bool)
    assert _get_peak_rss_kb() >= 0


def test_binary_feature_series_fallback_for_unhashable_values() -> None:
    series = pd.Series([["a"], ["b"], ["a"], None], dtype=object)
    assert _is_binary_feature_series(series)
    assert not _is_binary_feature_series(pd.Series([None, None], dtype=object))


def test_native_augmented_pair_transform_block_marginal_path() -> None:
    original_core = clf_mod._core
    original_core_available = clf_mod._CORE_AVAILABLE

    def score_pair_candidates(X_selected, y_codes, names):
        assert X_selected.shape[1] == 3
        assert y_codes.shape[0] == X_selected.shape[0]
        return [
            {
                "name": "a__product__b",
                "inputs": ["a", "b"],
                "operation": "product",
                "formula": "a * b",
                "transform_ig": 0.30,
                "reference_raw_value": 0.0,
                "eligible_count": 6,
                "eligible_rate": 1.0,
                "missing_pair_rate": 0.0,
                "transform_bin_edges": [0.0, 1.0, 2.0],
            },
            {
                "name": "b__sum__c",
                "inputs": ["b", "c"],
                "operation": "sum",
                "formula": "b + c",
                "transform_ig": 0.20,
                "reference_raw_value": 0.0,
                "eligible_count": 6,
                "eligible_rate": 1.0,
                "missing_pair_rate": 0.0,
                "transform_bin_edges": [0.0, 1.0, 2.0],
            },
        ]

    def transform_pair_features(X_selected, left, right, ops, refs, means, scales):
        out = []
        for li, ri, op, ref, mean, scale in zip(left, right, ops, refs, means, scales):
            lhs = X_selected[:, int(li)]
            rhs = X_selected[:, int(ri)]
            if int(op) == 0:
                raw = lhs * rhs
            elif int(op) == 2:
                raw = lhs + rhs
            else:
                raw = np.abs(lhs - rhs)
            out.append((raw - float(ref) - float(mean)) / float(scale))
        return np.column_stack(out) if out else np.zeros((X_selected.shape[0], 0))

    clf_mod._core = SimpleNamespace(
        score_pair_candidates=score_pair_candidates,
        transform_pair_features=transform_pair_features,
    )
    clf_mod._CORE_AVAILABLE = True
    try:
        X = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "b": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
                "c": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            }
        )
        block = NativeAugmentedPairTransformBlock(
            augmented_pair_mode="marginal_ig",
            aug_feature_size=3,
            max_pair_features=3,
            budget_topK=2,
            min_source_ig=0.0,
        )
        block.fit(
            X,
            np.array([0, 1, 0, 1, 0, 1]),
            ig_scores={"a": {"best": 0.4}, "b": 0.3, "c": 0.2},
            bin_edges={"a": [1.0, 3.0, 6.0], "b": [2.0, 3.0, 5.0], "c": [0.0, 1.0]},
            numeric_cols=["a", "b", "c"],
            full_feature_names=["a", "b", "c"],
        )
        assert block.candidate_count_ == 2
        assert block.feature_names_ == ["a__product__b", "b__sum__c"]
        transformed = block.transform(X)
        assert transformed.shape == (6, 2)
        catalog = block.augmented_pair_transforms_
        assert catalog[0]["selected_by"] == "marginal_ig"
        assert catalog[0]["source_bin_edges"]["a"] == [1.0, 3.0, 6.0]

        empty = NativeAugmentedPairTransformBlock(
            augmented_pair_mode="marginal_ig",
            min_source_ig=10.0,
        )
        empty.fit(X, np.array([0, 1, 0, 1, 0, 1]), {}, {}, ["a", "b"])
        assert empty.transform(X).shape == (6, 0)
        empty.kept_specs_ = [{"inputs": ["a", "b"], "operation": "unknown"}]
        empty.selected_aug_features_ = ["a", "b"]
        with pytest.raises(Exception, match="Unknown augmented-pair operation"):
            empty._pair_index_arrays()
    finally:
        clf_mod._core = original_core
        clf_mod._CORE_AVAILABLE = original_core_available


if __name__ == "__main__":
    sys.exit(main())


def test_extra_features_script_entrypoint_is_available() -> None:
    assert callable(main)
