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

from __future__ import annotations

import json
import os
import pickle
import struct
import sys
import types
import zipfile

import numpy as np
import pytest


class PickleFriendlyEstimator:
    def __init__(self, value=1):
        self.value = value
        self.n_features_in_ = 2


class PickleFriendlyDoubleTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X) * 2



class TestBinningHelperEdges:
    def test_entropy_empty_and_binning_fallbacks(self):
        from hugiml._binning import (
            _apply_edges,
            _entropy,
            _information_gain,
            _quantile_edges,
            _select_b,
        )

        assert _entropy(np.array([])) == 0.0
        x = np.ones(6)
        y = np.array([0, 1, 0, 1, 0, 1])
        assert _information_gain(x, y, 3) == 0.0
        chosen, scores = _select_b(x, y, [2, 4], min_marginal_gain_ratio=0.05)
        assert chosen == 2
        assert scores == {2: 0.0, 4: 0.0}

        np.testing.assert_allclose(_quantile_edges(np.array([np.nan, np.inf]), 4), [0.0, 1.0])
        constant_edges = _quantile_edges(np.array([3.0, 3.0, np.nan]), 4)
        assert constant_edges[0] == 3.0
        assert constant_edges[1] > constant_edges[0]

        labels = _apply_edges(np.array([-10.0, 0.25, 2.0, np.nan, np.inf]), np.array([0.0, 1.0]))
        assert labels[0] == "[0,1)"
        assert labels[1] == "[0,1)"
        assert labels[2] == "[0,1)"
        assert labels[3] is np.nan
        assert labels[4] is np.nan


class TestCalibrationEdges:
    def test_result_summary_and_to_dict(self):
        from hugiml.calibration import CalibrationResult

        result = CalibrationResult(
            ece=0.1,
            mce=0.2,
            brier_score=0.3,
            brier_reliability=0.01,
            brier_resolution=0.02,
            brier_uncertainty=0.25,
            n_bins=2,
            bin_confidences=[0.25, 0.75],
            bin_accuracies=[0.2, 0.8],
            bin_counts=[5, 5],
        )
        assert "Calibration Summary" in result.summary()
        as_dict = result.to_dict()
        assert as_dict["ece"] == 0.1
        assert as_dict["bin_counts"] == [5, 5]

    def test_invalid_inputs_and_empty_reliability_paths(self):
        from hugiml.calibration import (
            brier_decomposition,
            evaluate_calibration,
            reliability_diagram_data,
        )

        with pytest.raises(ValueError, match="n_bins"):
            evaluate_calibration(np.array([0, 1]), np.array([0.2, 0.8]), n_bins=1)
        with pytest.raises(ValueError, match="strategy"):
            evaluate_calibration(np.array([0, 1]), np.array([0.2, 0.8]), strategy="bad")
        with pytest.raises(ValueError, match="outside"):
            evaluate_calibration(np.array([0, 1]), np.array([0.2, np.nan]))

        assert brier_decomposition(np.array([]), np.array([])) == (0.0, 0.0, 0.0)
        confs, accs, counts = reliability_diagram_data(
            np.array([0, 1, 1]), np.array([0.2, 0.6, 0.9]), n_bins=3
        )
        assert len(confs) == len(accs) == len(counts)
        assert sum(counts) == 3

    def test_multiclass_and_quantile_calibration(self):
        from hugiml.calibration import evaluate_calibration

        y_true = np.array([0, 1, 2, 1])
        y_proba = np.array(
            [
                [0.80, 0.10, 0.10],
                [0.20, 0.70, 0.10],
                [0.25, 0.25, 0.50],
                [0.60, 0.30, 0.10],
            ]
        )
        result = evaluate_calibration(y_true, y_proba, n_bins=3, strategy="quantile")
        assert result.n_bins == 3
        assert result.brier_score >= 0.0
        assert sum(result.bin_counts) == 4


class TestGovernanceArtifacts:
    def test_model_card_governance_and_audit_serialization(self, tmp_path):
        from hugiml.governance import AuditArtifact, GovernanceMetadata, ModelCard

        card = ModelCard(
            model_id="model-a",
            intended_use="classification",
            hyperparameters={"B": 4},
            performance_metrics={"accuracy": 0.9},
            top_patterns=["a=1", "b=2"],
            limitations=["sample limitation"],
        )
        assert "model-a" in card.to_json()
        assert "classification" in card.to_markdown()
        card_path = tmp_path / "card.md"
        card.save(str(card_path), fmt="md")
        assert card_path.read_text(encoding="utf-8").startswith("# Model Card")

        gov = GovernanceMetadata(model_id="model-a", owner="team", tags=["prod"])
        assert json.loads(gov.to_json())["owner"] == "team"

        artifact = AuditArtifact(model_id="model-a", governance=gov.to_dict())
        audit_path = tmp_path / "audit.json"
        artifact.save(str(audit_path))
        assert json.loads(audit_path.read_text(encoding="utf-8"))["model_id"] == "model-a"

    def test_generate_model_card_fallback_paths(self):
        from hugiml.governance import generate_model_card

        class Meta:
            config = {"B": 3}
            n_patterns = 2
            n_compound = 1

        class Classifier:
            fit_metadata_ = Meta()

            def feature_importances(self):
                raise RuntimeError("not available")

            def get_hug_features(self):
                return ["x=[0,1)", "y=A"]

        card = generate_model_card(
            Classifier(),
            "fallback-model",
            performance_metrics={"auc": 0.8},
            ethical_considerations="reviewed",
        )
        assert card.hyperparameters == {"B": 3}
        assert card.n_patterns == 2
        assert card.top_patterns == ["x=[0,1)", "y=A"]
        assert card.performance_metrics["auc"] == 0.8

    def test_package_audit_artifacts_handles_optional_failures(self, tmp_path):
        from hugiml.governance import GovernanceMetadata, package_audit_artifacts

        class BadOptional:
            def to_dict(self):
                raise RuntimeError("unavailable")

            def to_json(self):
                raise RuntimeError("unavailable")

        class Classifier:
            fit_metadata_ = None

            def feature_importances(self):
                raise RuntimeError("not available")

            def get_pattern_info(self):
                raise RuntimeError("not available")

            def get_hug_features(self):
                raise RuntimeError("not available")

        manifest = package_audit_artifacts(
            Classifier(),
            "audit-model",
            str(tmp_path),
            governance=GovernanceMetadata(model_id="audit-model"),
            calibration_result=BadOptional(),
            explainability_report=BadOptional(),
        )
        data = json.loads((tmp_path / "audit_manifest.json").read_text(encoding="utf-8"))
        assert manifest.endswith("audit_manifest.json")
        assert data["training_hash"] == "unavailable"
        assert data["pattern_info"] == []


class TestTelemetryEnabledAndErrorPaths:
    def test_enabled_tracer_records_attributes_and_attribute_errors(self):
        import hugiml.telemetry as telemetry

        calls: list[tuple[str, object]] = []

        class Span:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def set_attribute(self, key, value):
                calls.append((key, value))
                if key == "bad":
                    raise RuntimeError("attribute rejected")

            def record_exception(self, exc):
                calls.append(("exception", type(exc).__name__))

        class Tracer:
            def start_as_current_span(self, name):
                calls.append(("span", name))
                return Span()

        old_enabled = telemetry._OTEL_ENABLED
        old_tracer = telemetry.HUGIMLTracer._tracer
        try:
            telemetry._OTEL_ENABLED = True
            telemetry.HUGIMLTracer._tracer = Tracer()
            with telemetry.HUGIMLTracer.span("custom", {"ok": 1, "bad": 2}) as span:
                span.record_exception(ValueError("x"))
        finally:
            telemetry._OTEL_ENABLED = old_enabled
            telemetry.HUGIMLTracer._tracer = old_tracer
        assert ("span", "custom") in calls
        assert ("ok", 1) in calls
        assert ("exception", "ValueError") in calls

    def test_enabled_metrics_use_counter_histogram_and_gauge(self):
        import hugiml.telemetry as telemetry

        events: list[tuple[str, object]] = []

        class Metric:
            def __init__(self, *args, **kwargs):
                events.append(("create", args[0]))

            def labels(self, **labels):
                events.append(("labels", tuple(sorted(labels))))
                return self

            def inc(self, value):
                events.append(("inc", value))

            def observe(self, value):
                events.append(("observe", value))

            def set(self, value):
                events.append(("set", value))

        fake_module = types.SimpleNamespace(Counter=Metric, Gauge=Metric, Histogram=Metric)
        old_module = sys.modules.get("prometheus_client")
        old_enabled = telemetry._PROM_ENABLED
        old_initialized = telemetry.HUGIMLMetrics._initialized
        old_counter = telemetry.HUGIMLMetrics._predictions_total
        old_latency = telemetry.HUGIMLMetrics._prediction_latency
        old_confidence = telemetry.HUGIMLMetrics._confidence_mean
        old_drift = telemetry.HUGIMLMetrics._drift_psi
        try:
            sys.modules["prometheus_client"] = fake_module
            telemetry._PROM_ENABLED = True
            telemetry.HUGIMLMetrics._initialized = False
            telemetry.HUGIMLMetrics.record_prediction("m", 3, 0.2, 0.7, success=False)
            telemetry.HUGIMLMetrics.record_drift("m", {"feature": 0.12})
        finally:
            if old_module is None:
                sys.modules.pop("prometheus_client", None)
            else:
                sys.modules["prometheus_client"] = old_module
            telemetry._PROM_ENABLED = old_enabled
            telemetry.HUGIMLMetrics._initialized = old_initialized
            telemetry.HUGIMLMetrics._predictions_total = old_counter
            telemetry.HUGIMLMetrics._prediction_latency = old_latency
            telemetry.HUGIMLMetrics._confidence_mean = old_confidence
            telemetry.HUGIMLMetrics._drift_psi = old_drift

        assert ("inc", 3) in events
        assert ("observe", 0.2) in events
        assert ("set", 0.7) in events
        assert ("set", 0.12) in events

    def test_instrumented_classifier_records_success_and_error(self):
        import hugiml.telemetry as telemetry

        records: list[tuple[str, int, bool]] = []
        exceptions: list[str] = []

        class Span:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def set_attribute(self, *args):
                return None

            def record_exception(self, exc):
                exceptions.append(type(exc).__name__)

        class Classifier:
            patterns_ = ["p1"]

            def __init__(self):
                self.raise_next = False

            def predict_proba(self, X):
                if self.raise_next:
                    raise ValueError("bad input")
                return np.array([[0.2, 0.8], [0.6, 0.4]])[: len(X)]

            def predict(self, X):
                return np.array([1] * len(X))

        def record_prediction(model_id, n_samples, latency_s, mean_confidence, success=True):
            records.append((model_id, n_samples, success))

        old_span = telemetry.HUGIMLTracer.span
        old_record = telemetry.HUGIMLMetrics.record_prediction
        try:
            telemetry.HUGIMLTracer.span = lambda *args, **kwargs: Span()
            telemetry.HUGIMLMetrics.record_prediction = record_prediction
            clf = telemetry.instrument_classifier(Classifier(), model_id="wrapped")
            np.testing.assert_array_equal(clf.predict([1, 2, 3]), np.array([1, 1, 1]))
            assert clf.predict_proba([1, 2]).shape == (2, 2)
            clf.raise_next = True
            with pytest.raises(ValueError):
                clf.predict_proba([1])
        finally:
            telemetry.HUGIMLTracer.span = old_span
            telemetry.HUGIMLMetrics.record_prediction = old_record
        assert ("wrapped", 2, True) in records
        assert ("wrapped", 0, False) in records
        assert exceptions == ["ValueError"]


class TestSerializationHelperEdges:
    def test_hmac_key_validation_and_json_conversions(self):
        import hugiml.serialization as ser
        from hugiml.exceptions import HUGIMLSerializationError

        old_require = os.environ.get("HUGIML_REQUIRE_MODEL_HMAC")
        old_key = os.environ.get("HUGIML_MODEL_HMAC_KEY")
        try:
            os.environ["HUGIML_REQUIRE_MODEL_HMAC"] = "yes"
            assert ser._require_hmac() is True

            os.environ["HUGIML_MODEL_HMAC_KEY"] = "not-hex"
            with pytest.raises(HUGIMLSerializationError, match="hex"):
                ser._get_hmac_key()
            os.environ["HUGIML_MODEL_HMAC_KEY"] = "00"
            with pytest.raises(HUGIMLSerializationError, match="at least"):
                ser._get_hmac_key()
            os.environ["HUGIML_MODEL_HMAC_KEY"] = "ab" * 16
            assert ser._get_hmac_key() == bytes.fromhex("ab" * 16)
        finally:
            if old_require is None:
                os.environ.pop("HUGIML_REQUIRE_MODEL_HMAC", None)
            else:
                os.environ["HUGIML_REQUIRE_MODEL_HMAC"] = old_require
            if old_key is None:
                os.environ.pop("HUGIML_MODEL_HMAC_KEY", None)
            else:
                os.environ["HUGIML_MODEL_HMAC_KEY"] = old_key

        payload = ser._json_dumps(
            {
                "i": np.int64(4),
                "f": np.float64(0.25),
                "a": np.array([1, 2]),
                "b": np.bool_(True),
            }
        )
        assert json.loads(payload) == {"i": 4, "f": 0.25, "a": [1, 2], "b": True}
        with pytest.raises(TypeError):
            ser._json_dumps({"bad": object()})

    def test_estimator_fallback_and_pipeline_roundtrip(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        import hugiml.serialization as ser

        old_safe_modules = ser._SAFE_MODULES
        ser._SAFE_MODULES = (*ser._SAFE_MODULES, __name__)
        try:
            est = PickleFriendlyEstimator(value=7)
            config, arrays = ser._serialize_estimator(est)
            assert config["_pickle_fallback"] is True
            restored = ser._deserialize_estimator(config, arrays)
            assert restored.value == 7
        finally:
            ser._SAFE_MODULES = old_safe_modules

        X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
        y = np.array([0, 0, 1, 1])
        pipe = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
        pipe.fit(X, y)
        pipe_config, pipe_arrays = ser._serialize_estimator(pipe)
        assert pipe_config["class"] == "sklearn.pipeline.Pipeline"
        pipe2 = ser._deserialize_estimator(pipe_config, pipe_arrays)
        np.testing.assert_allclose(pipe.predict_proba(X), pipe2.predict_proba(X))

        pipe_fallback = Pipeline(
            [("double", PickleFriendlyDoubleTransformer()), ("lr", LogisticRegression(max_iter=1000))]
        )
        pipe_fallback.fit(X, y)
        old_safe_modules = ser._SAFE_MODULES
        ser._SAFE_MODULES = (*ser._SAFE_MODULES, __name__)
        try:
            fallback_config, fallback_arrays = ser._serialize_pipeline(pipe_fallback)
            assert fallback_config["steps"][0]["estimator"]["_pickle_fallback"] is True
            pipe3 = ser._deserialize_pipeline(fallback_config, fallback_arrays)
            np.testing.assert_allclose(pipe_fallback.predict_proba(X), pipe3.predict_proba(X))
        finally:
            ser._SAFE_MODULES = old_safe_modules

    def test_deserialization_rejects_unknown_estimator_and_unsafe_pickle(self):
        import hugiml.serialization as ser
        from hugiml.exceptions import HUGIMLSerializationError

        with pytest.raises(HUGIMLSerializationError, match="Cannot deserialize"):
            ser._deserialize_estimator({"class": "unknown.Estimator"}, {})

        payload = pickle.dumps(types.SimpleNamespace(value=1), protocol=5)
        with pytest.raises(pickle.UnpicklingError, match="not allowed"):
            ser._safe_unpickle(payload)

    def test_load_model_archive_errors_and_sbom_output(self, tmp_path):
        import hugiml.serialization as ser
        from hugiml.exceptions import HUGIMLSerializationError, HUGIMLVersionError

        incomplete = tmp_path / "incomplete.hugiml"
        with zipfile.ZipFile(incomplete, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"schema_version": ser.MODEL_SCHEMA_VERSION}))
        with pytest.raises(HUGIMLSerializationError, match="incomplete"):
            ser.load_model(incomplete)

        old_schema = tmp_path / "old_schema.hugiml"
        required = {
            "manifest.json": json.dumps({"schema_version": 0}),
            "clf_init.json": "{}",
            "clf_fit.json": "{}",
            "patterns.json": "[]",
            "arrays.npz": ser._npz_bytes(classes_=np.array([0, 1])),
            "td_config.json": "{}",
            "td_arrays.npz": ser._npz_bytes(),
            "estimator.json": "{}",
            "estimator_arrays.npz": ser._npz_bytes(),
            "hmac.sig": "0" * 64,
        }
        with zipfile.ZipFile(old_schema, "w") as zf:
            for name, content in required.items():
                zf.writestr(name, content)
        with pytest.raises(HUGIMLVersionError, match="too old"):
            ser.load_model(old_schema)

        legacy = tmp_path / "legacy_bad.hugiml"
        legacy.write_bytes(b"BAD!" + struct.pack("<I", 1) + (b"\x00" * 32))
        with pytest.raises(HUGIMLSerializationError, match="legacy"):
            ser._load_legacy(legacy)

        sbom_path = tmp_path / "sbom.json"
        sbom = ser.generate_sbom(str(sbom_path))
        assert sbom["bomFormat"] == "CycloneDX-lite"
        assert sbom_path.exists()


class TestClassifierPrivateHelpers:
    def test_information_gain_and_quantile_code_helpers(self):
        from hugiml.classifier import (
            _best_ig_score,
            _continuous_to_quantile_codes,
            _dense_full_csr,
            _entropy_from_counts,
            _information_gain_from_codes,
        )

        assert _best_ig_score({"a": "bad", "b": np.nan}) == 0.0
        assert _best_ig_score({"a": 0.1, "b": 0.4}) == 0.4
        assert _best_ig_score("bad") == 0.0
        assert _best_ig_score(np.inf) == 0.0

        assert _entropy_from_counts(np.array([0, 0])) == 0.0
        assert _information_gain_from_codes(
            np.array([-1, -1]), np.array([0, 1]), n_classes=2
        ) == 0.0
        assert _information_gain_from_codes(
            np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), n_classes=2
        ) > 0.9

        empty_csr = _dense_full_csr(np.zeros((3, 0)))
        assert empty_csr.shape == (3, 0)
        dense_csr = _dense_full_csr(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert dense_csr.nnz == 4
        np.testing.assert_allclose(dense_csr.toarray(), [[1.0, 2.0], [3.0, 4.0]])

        all_missing_codes = _continuous_to_quantile_codes(np.array([np.nan, np.inf]), max_bins=4)
        np.testing.assert_array_equal(all_missing_codes, np.array([-1, -1]))
        low_card_codes = _continuous_to_quantile_codes(np.array([1.0, 1.0, 2.0, np.nan]), max_bins=4)
        np.testing.assert_array_equal(low_card_codes, np.array([0, 0, 1, -1]))
        high_card_codes = _continuous_to_quantile_codes(np.arange(20.0), max_bins=4)
        assert high_card_codes.min() == 0
        assert high_card_codes.max() <= 4

    def test_augmented_pair_empty_selection_and_catalog(self):
        from hugiml.classifier import NativeAugmentedPairTransformBlock

        block = NativeAugmentedPairTransformBlock(
            aug_feature_size=2,
            budget_topK=-1,
            unbounded_cap=3,
            augmented_pair_mode="marginal_ig",
        )
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        block.fit(
            X,
            y,
            ig_scores={"a": {2: 0.0}, "b": {2: 0.0}},
            bin_edges={},
            numeric_cols=["a", "b"],
            full_feature_names=["a", "b"],
        )
        assert block.budget_topK == 3
        assert block.selected_aug_features_ == []
        assert block.transform(X).shape == (2, 0)
        assert block.augmented_pair_transforms_ == []

        block.selected_aug_features_ = ["a", "b"]
        block.selected_aug_scores_ = {"a": 0.5, "b": 0.4}
        block.source_observed_medians_ = {"a": 1.0, "b": 2.0}
        block.input_bin_edges_ = {"a": [0.0, 1.0], "b": [1.0, 2.0]}
        block.kept_specs_ = [
            {
                "name": "a*b",
                "operation": "product",
                "inputs": ["a", "b"],
                "formula": "a * b",
                "reference_raw_value": 2.0,
                "eligible_count": 2,
                "eligible_rate": 1.0,
                "missing_pair_rate": 0.0,
                "transform_ig": 0.2,
                "transform_bin_edges": [0.0, 2.0, 4.0],
            }
        ]
        block.scaler_mean_ = np.array([2.0])
        block.scaler_scale_ = np.array([0.5])
        block.candidate_count_ = 1
        catalog = block._build_catalog()
        assert catalog[0]["standardization"] == {"mean": 2.0, "scale": 0.5}
        assert catalog[0]["source_ig"] == {"a": 0.5, "b": 0.4}
        assert catalog[0]["source_bin_edges"] == {"a": [0.0, 1.0], "b": [1.0, 2.0]}

        block.kept_specs_[0]["operation"] = "unknown"
        with pytest.raises(Exception, match="Unknown augmented-pair"):
            block._pair_index_arrays()

    def test_classifier_adaptive_binning_direct_paths(self):
        from hugiml.classifier import HUGIMLClassifierNative

        rng = np.random.default_rng(123)
        X = np.column_stack([np.linspace(0.0, 1.0, 24), rng.normal(size=24)])
        X[3, 1] = np.nan
        y = np.array([0, 1] * 12)
        clf = HUGIMLClassifierNative(B=3, adaptive_binning=True, b_candidates=[2, 3])
        clf.feature_names_in_ = ["a", "b"]
        clf.cat_cols_mask_ = np.array([False, False])
        clf.is_int_mask_ = np.array([False, False])
        clf.n_jobs = 1

        pre = clf._apply_adaptive_binning(X, y)
        assert list(pre.columns) == ["a", "b"]
        assert set(clf._bin_edges_) == {"a", "b"}
        assert clf.cat_cols_mask_.tolist() == [False, False]
        assert clf.is_int_mask_.tolist() == [True, True]
        assert any(key.startswith("a=[0.000") for key in clf._adaptive_code_label_map_)
        assert np.isnan(pre.loc[3, "b"])

        out = clf._prebin_for_predict(np.array([[0.2, 0.4], [2.0, np.nan]]))
        assert out.shape == (2, 2)
        assert np.isfinite(out[0, 0])
        assert np.isnan(out[1, 1])

        frame_out = clf._prebin_for_predict(pre[["a", "b"]].copy())
        assert list(frame_out.columns) == ["a", "b"]
        assert np.isnan(frame_out.loc[3, "b"])

        clf._adaptive_precoded_features_ = set()
        labeled = clf._prebin_for_predict(np.array([[0.2, 0.4]]))
        assert isinstance(labeled.loc[0, "a"], str)
