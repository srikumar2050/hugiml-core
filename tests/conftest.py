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

"""Shared pytest fixtures for the hugiml-core test suite.

Datasets
--------
Two real-world datasets are used throughout the suite:

* **German Credit** (german.data) — 1000 samples, 20 mixed features,
  binary creditworthiness target (1 = Good, 2 = Bad).
* **HELOC** (heloc.csv) — ~10 000 samples, 23 numeric features,
  binary repayment performance target (Good / Bad).

Both datasets live under ``tests/`` and are loaded once per session via
session-scoped fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
GERMAN_PATH = TESTS_DIR / "german.data"
HELOC_PATH = TESTS_DIR / "heloc.csv"

# German Credit column names (UCI ML Repository specification)
GERMAN_COLS = [
    "checking_acct",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "target",
]


# ---------------------------------------------------------------------------
# German Credit dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def german_raw() -> pd.DataFrame:
    """Full German Credit DataFrame with a binary target column (0/1)."""
    df = pd.read_csv(GERMAN_PATH, sep=" ", header=None, names=GERMAN_COLS)
    # Original coding: 1 = Good, 2 = Bad → recode to 0/1
    df["target"] = (df["target"] == 2).astype(int)
    return df


@pytest.fixture(scope="session")
def german_Xy(german_raw) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix and label series for German Credit."""
    X = german_raw.drop(columns=["target"])
    y = german_raw["target"]
    return X, y


@pytest.fixture(scope="session")
def german_split(german_Xy):
    """80/20 stratified train/test split for German Credit."""
    from sklearn.model_selection import train_test_split

    X, y = german_Xy
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ---------------------------------------------------------------------------
# HELOC dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def heloc_raw() -> pd.DataFrame:
    """Full HELOC DataFrame with a binary target column (0/1)."""
    df = pd.read_csv(HELOC_PATH)
    df["target"] = (df["RiskPerformance"] == "Bad").astype(int)
    df = df.drop(columns=["RiskPerformance"])
    return df


@pytest.fixture(scope="session")
def heloc_Xy(heloc_raw) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix and label series for HELOC."""
    X = heloc_raw.drop(columns=["target"])
    y = heloc_raw["target"]
    return X, y


@pytest.fixture(scope="session")
def heloc_split(heloc_Xy):
    """80/20 stratified train/test split for HELOC."""
    from sklearn.model_selection import train_test_split

    X, y = heloc_Xy
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ---------------------------------------------------------------------------
# Small synthetic dataset (fast smoke tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_Xy():
    """Tiny 200-sample synthetic dataset for fast unit tests.

    Three integer columns, two float columns, one categorical column,
    binary target.  Deterministic seed for reproducibility.
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "int_a": rng.integers(0, 10, n),
            "int_b": rng.integers(0, 5, n),
            "int_c": rng.integers(1, 20, n),
            "float_x": rng.uniform(0, 1, n),
            "float_y": rng.uniform(-2, 2, n),
            "cat_z": rng.choice(["A", "B", "C"], n),
        }
    )
    # Target correlated with int_a and float_x
    logits = 0.3 * df["int_a"] + 0.5 * df["float_x"] - 0.1 * df["int_b"]
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = pd.Series(rng.binomial(1, prob).astype(int), name="target")
    return df, y


@pytest.fixture(scope="session")
def synthetic_split(synthetic_Xy):
    """Stratified 80/20 split of the synthetic dataset."""
    from sklearn.model_selection import train_test_split

    X, y = synthetic_Xy
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


# ---------------------------------------------------------------------------
# Pre-fitted classifiers (session-scoped to amortise fit cost)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fitted_clf_synthetic(synthetic_split):
    """HUGIMLClassifierNative fitted on the synthetic training split."""
    from hugiml import HUGIMLClassifierNative

    X_tr, X_te, y_tr, y_te = synthetic_split
    clf = HUGIMLClassifierNative(B=5, L=1, G=1e-3)
    X_tr_p, y_tr_p = clf.prepareXy(X_tr, y_tr)
    # Re-split after prepareXy to keep the fixture self-contained
    from sklearn.model_selection import train_test_split

    X_f, _, y_f, _ = train_test_split(
        X_tr_p, y_tr_p, test_size=0.1, random_state=0, stratify=y_tr_p
    )
    clf.fit(X_f, y_f)
    return clf, X_te, y_te


@pytest.fixture(scope="session")
def fitted_clf_german(german_Xy):
    """HUGIMLClassifierNative fitted on German Credit training data."""
    from sklearn.model_selection import train_test_split

    from hugiml import HUGIMLClassifierNative

    X, y = german_Xy
    clf = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
    X_p, y_p = clf.prepareXy(X, y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_p, y_p, test_size=0.2, random_state=42, stratify=y_p
    )
    clf.fit(X_tr, y_tr)
    return clf, X_te, y_te


# ---------------------------------------------------------------------------
# Temp directory
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_path_model(tmp_path) -> Path:
    """Convenience fixture: returns a path to a temporary model file."""
    return tmp_path / "model.hugiml"


# ---------------------------------------------------------------------------
# Extension availability guard
# ---------------------------------------------------------------------------


def _core_available() -> bool:
    """Return True when the _hugiml_core native extension is importable."""
    try:
        import _hugiml_core  # noqa: F401

        return True
    except ImportError:
        return False


#: Module-level mark: apply to any test that calls HUGIMLClassifierNative.fit().
#: Tests decorated with this skip gracefully when the extension is not built.
requires_extension = pytest.mark.skipif(
    not _core_available(),
    reason=(
        "_hugiml_core native extension not built.\n"
        "Build with: HUGIML_FAST_BUILD=1 pip install -e . --no-build-isolation"
    ),
)
