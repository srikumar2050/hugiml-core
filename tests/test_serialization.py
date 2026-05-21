# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for serialization.py — save/load round-trip, SBOM generation,
schema version checks, authenticated signing, and v3 non-pickle format.
"""

from __future__ import annotations

import json
import os
import zipfile

import numpy as np
import pytest

from hugiml import HUGIMLClassifierNative
from hugiml.serialization import (
    MODEL_SCHEMA_VERSION,
    generate_sbom,
    load_model,
    save_model,
)

# =============================================================================
# Helpers
# =============================================================================


def _assert_predictions_match(clf1, clf2, X_te):
    p1 = clf1.predict(X_te)
    p2 = clf2.predict(X_te)
    np.testing.assert_array_equal(p1, p2)


def _assert_proba_match(clf1, clf2, X_te):
    p1 = clf1.predict_proba(X_te)
    p2 = clf2.predict_proba(X_te)
    np.testing.assert_allclose(p1, p2, atol=1e-10)


# =============================================================================
# Save / Load round-trip
# =============================================================================


class TestSaveLoadRoundTrip:
    def test_save_creates_file(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_saved_file_is_zip(self, fitted_clf_synthetic, tmp_path):
        """v3 format must be a valid ZIP archive."""
        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        assert zipfile.is_zipfile(out), "saved model is not a valid ZIP archive"

    def test_zip_contains_required_members(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        with zipfile.ZipFile(out, "r") as zf:
            names = set(zf.namelist())
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
        missing = required - names
        assert not missing, f"Missing archive members: {sorted(missing)}"

    def test_manifest_schema_version(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        with zipfile.ZipFile(out, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
        assert manifest["schema_version"] == MODEL_SCHEMA_VERSION
        assert manifest["format_version"] == MODEL_SCHEMA_VERSION

    def test_load_returns_classifier(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        clf2 = load_model(out)
        assert isinstance(clf2, HUGIMLClassifierNative)

    def test_predictions_match_after_roundtrip(self, fitted_clf_synthetic, tmp_path):
        clf, X_te, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        clf2 = load_model(out)
        _assert_predictions_match(clf, clf2, X_te)

    def test_predict_proba_match_after_roundtrip(self, fitted_clf_synthetic, tmp_path):
        clf, X_te, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        clf2 = load_model(out)
        _assert_proba_match(clf, clf2, X_te)

    def test_hug_features_preserved(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        clf2 = load_model(out)
        assert clf.get_hug_features() == clf2.get_hug_features()

    def test_n_features_in_preserved(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        clf2 = load_model(out)
        assert clf2.n_features_in_ == clf.n_features_in_

    def test_classes_preserved(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        clf2 = load_model(out)
        np.testing.assert_array_equal(clf2.classes_, clf.classes_)

    def test_patterns_count_preserved(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        clf2 = load_model(out)
        assert len(clf2.patterns_) == len(clf.patterns_)

    def test_save_via_method(self, fitted_clf_synthetic, tmp_path):
        """Classifier.save_model() must write a valid v3 archive."""
        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "via_method.hugiml"
        clf.save_model(out)
        assert zipfile.is_zipfile(out)
        clf2 = load_model(out)
        assert isinstance(clf2, HUGIMLClassifierNative)

    def test_path_string_accepted(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        out = str(tmp_path / "model_str.hugiml")
        save_model(clf, out)
        clf2 = load_model(out)
        assert isinstance(clf2, HUGIMLClassifierNative)

    def test_overwrite_existing_file(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        save_model(clf, out)  # second save must not raise
        assert zipfile.is_zipfile(out)

    def test_expected_type_check_passes(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "typed.hugiml"
        save_model(clf, out)
        clf2 = load_model(out, expected_type=HUGIMLClassifierNative)
        assert isinstance(clf2, HUGIMLClassifierNative)

    def test_expected_type_check_fails(self, fitted_clf_synthetic, tmp_path):
        from hugiml.exceptions import HUGIMLSerializationError

        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "typed.hugiml"
        save_model(clf, out)
        with pytest.raises(HUGIMLSerializationError, match="expected"):
            load_model(out, expected_type=str)


# =============================================================================
# No-pickle guarantee
# =============================================================================


class TestNoPickle:
    def test_archive_has_no_pickle_magic(self, fitted_clf_synthetic, tmp_path):
        """The ZIP archive members must not contain raw pickle data (\\x80\\x05)."""
        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        with zipfile.ZipFile(out, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".npz") or name == "hmac.sig":
                    continue  # npz is binary but not pickle
                raw = zf.read(name)
                # Pickle protocol 5 starts with \x80\x05
                assert b"\x80\x05" not in raw, (
                    f"Pickle magic bytes found in {name!r}; "
                    "the v3 format must not contain pickle data in JSON/config files."
                )

    def test_arrays_npz_uses_numpy_format(self, fitted_clf_synthetic, tmp_path):
        """arrays.npz must load without allow_pickle=True."""
        import io

        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "model.hugiml"
        save_model(clf, out)
        with zipfile.ZipFile(out, "r") as zf:
            raw = zf.read("arrays.npz")
        loaded = np.load(io.BytesIO(raw), allow_pickle=False)
        assert "classes_" in loaded


# =============================================================================
# HMAC signing
# =============================================================================


class TestHMACSigning:
    def test_unsigned_model_loads_without_key(self, fitted_clf_synthetic, tmp_path):
        """Models saved without a key load fine when no key is configured."""
        clf, X_te, _ = fitted_clf_synthetic
        out = tmp_path / "unsigned.hugiml"
        save_model(clf, out)  # no HUGIML_MODEL_HMAC_KEY set
        clf2 = load_model(out)
        assert isinstance(clf2, HUGIMLClassifierNative)

    def test_signed_model_verifies_with_correct_key(self, fitted_clf_synthetic, tmp_path):
        clf, X_te, _ = fitted_clf_synthetic
        out = tmp_path / "signed.hugiml"
        key_hex = "deadbeef" * 8  # 32 bytes

        import unittest.mock as mock

        with mock.patch.dict(os.environ, {"HUGIML_MODEL_HMAC_KEY": key_hex}):
            save_model(clf, out)
            clf2 = load_model(out)
        assert isinstance(clf2, HUGIMLClassifierNative)

    def test_tampered_archive_fails_hmac(self, fitted_clf_synthetic, tmp_path):
        import unittest.mock as mock

        from hugiml.exceptions import HUGIMLSerializationError

        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "tampered.hugiml"
        key_hex = "cafebabe" * 8

        with mock.patch.dict(os.environ, {"HUGIML_MODEL_HMAC_KEY": key_hex}):
            save_model(clf, out)

        # Tamper: rewrite clf_fit.json with garbage

        members: dict[str, bytes] = {}
        with zipfile.ZipFile(out, "r") as zf:
            for name in zf.namelist():
                members[name] = zf.read(name)
        members["clf_fit.json"] = b'{"n_features_in_": 9999}'  # tampered
        tampered = tmp_path / "tampered2.hugiml"
        with zipfile.ZipFile(tampered, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)

        with mock.patch.dict(os.environ, {"HUGIML_MODEL_HMAC_KEY": key_hex}):
            with pytest.raises(HUGIMLSerializationError, match="HMAC"):
                load_model(tampered)

    def test_require_hmac_without_key_raises(self, fitted_clf_synthetic, tmp_path):
        import unittest.mock as mock

        from hugiml.exceptions import HUGIMLSerializationError

        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "nok.hugiml"
        save_model(clf, out)

        with mock.patch.dict(
            os.environ,
            {"HUGIML_REQUIRE_MODEL_HMAC": "true"},
            clear=False,
        ):
            # Ensure HUGIML_MODEL_HMAC_KEY is absent
            env_copy = {k: v for k, v in os.environ.items() if k != "HUGIML_MODEL_HMAC_KEY"}
            with mock.patch.dict(os.environ, env_copy, clear=True):
                with pytest.raises(HUGIMLSerializationError, match="HUGIML_REQUIRE_MODEL_HMAC"):
                    load_model(out)


# =============================================================================
# Error cases
# =============================================================================


class TestErrorCases:
    def test_load_nonexistent_file_raises(self, tmp_path):
        from hugiml.exceptions import HUGIMLSerializationError

        with pytest.raises(HUGIMLSerializationError):
            load_model(tmp_path / "does_not_exist.hugiml")

    def test_save_unfitted_raises(self, tmp_path):
        from hugiml.exceptions import HUGIMLSerializationError

        clf = HUGIMLClassifierNative()
        with pytest.raises(HUGIMLSerializationError, match="unfitted"):
            save_model(clf, tmp_path / "x.hugiml")

    def test_load_garbage_file_raises(self, tmp_path):
        from hugiml.exceptions import HUGIMLSerializationError

        p = tmp_path / "garbage.hugiml"
        p.write_bytes(b"not a zip or pickle at all")
        with pytest.raises(HUGIMLSerializationError):
            load_model(p)

    def test_load_raw_pickle_rejected(self, tmp_path):
        """A raw pickle file (not a HUG-IML envelope) must be rejected."""
        import pickle

        from hugiml.exceptions import HUGIMLSerializationError

        p = tmp_path / "evil.hugiml"
        p.write_bytes(pickle.dumps({"not_a_model": True}))
        with pytest.raises(HUGIMLSerializationError):
            load_model(p)

    def test_incomplete_archive_raises(self, fitted_clf_synthetic, tmp_path):
        """An archive with missing members must raise HUGIMLSerializationError."""
        from hugiml.exceptions import HUGIMLSerializationError

        clf, _, _ = fitted_clf_synthetic
        out = tmp_path / "complete.hugiml"
        save_model(clf, out)

        # Remove one required member
        members: dict[str, bytes] = {}
        with zipfile.ZipFile(out, "r") as zf:
            for name in zf.namelist():
                members[name] = zf.read(name)
        del members["patterns.json"]

        broken = tmp_path / "incomplete.hugiml"
        with zipfile.ZipFile(broken, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)

        with pytest.raises(HUGIMLSerializationError, match="incomplete"):
            load_model(broken)


# =============================================================================
# SBOM generation
# =============================================================================


class TestSBOMGeneration:
    def test_generate_sbom_returns_dict(self):
        sbom = generate_sbom()
        assert isinstance(sbom, dict)

    def test_sbom_has_required_keys(self):
        sbom = generate_sbom()
        assert "metadata" in sbom
        assert "components" in sbom

    def test_sbom_json_serialisable(self):
        sbom = generate_sbom()
        serialised = json.dumps(sbom)
        parsed = json.loads(serialised)
        assert isinstance(parsed, dict)

    def test_sbom_save_to_file(self, tmp_path):
        sbom = generate_sbom()
        out = tmp_path / "sbom.json"
        generate_sbom(output_path=str(out))
        assert out.exists()
        assert json.loads(out.read_text()) == sbom

    def test_sbom_contains_hugiml(self):
        sbom = generate_sbom()
        serialised = json.dumps(sbom).lower()
        assert "hugiml" in serialised

    def test_sbom_lists_numpy_scipy(self):
        sbom = generate_sbom()
        names = {c["name"] for c in sbom.get("components", [])}
        assert "numpy" in names
        assert "scipy" in names
