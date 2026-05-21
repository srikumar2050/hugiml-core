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

"""Tests for governance.py — ModelCard, GovernanceMetadata, AuditArtifact."""

from __future__ import annotations

import json
from pathlib import Path

from hugiml.governance import (
    AuditArtifact,
    GovernanceMetadata,
    ModelCard,
    generate_model_card,
    package_audit_artifacts,
)

# ---------------------------------------------------------------------------
# ModelCard
# ---------------------------------------------------------------------------


class TestModelCard:
    def _basic_card(self, fitted_clf_synthetic):
        clf, _, _ = fitted_clf_synthetic
        return generate_model_card(
            clf,
            model_id="test-model-v0.1",
            intended_use="Unit test model card",
            training_data_description="Synthetic 200-sample dataset",
        )

    def test_generate_model_card_returns_card(self, fitted_clf_synthetic):
        card = self._basic_card(fitted_clf_synthetic)
        assert isinstance(card, ModelCard)

    def test_model_card_to_json(self, fitted_clf_synthetic):
        card = self._basic_card(fitted_clf_synthetic)
        j = card.to_json()
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_model_card_to_markdown(self, fitted_clf_synthetic):
        card = self._basic_card(fitted_clf_synthetic)
        md = card.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_model_card_save_json(self, fitted_clf_synthetic, tmp_path):
        card = self._basic_card(fitted_clf_synthetic)
        out = tmp_path / "model_card.json"
        card.save(str(out), fmt="json")
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_model_card_save_markdown(self, fitted_clf_synthetic, tmp_path):
        card = self._basic_card(fitted_clf_synthetic)
        out = tmp_path / "model_card.md"
        card.save(str(out), fmt="md")
        assert out.exists()

    def test_model_card_contains_model_id(self, fitted_clf_synthetic):
        card = self._basic_card(fitted_clf_synthetic)
        j = card.to_json()
        assert "test-model-v0.1" in j

    def test_model_card_has_required_fields(self, fitted_clf_synthetic):
        card = self._basic_card(fitted_clf_synthetic)
        data = json.loads(card.to_json())
        # model_id is always present
        found_keys = {k.lower() for k in data}
        assert any("model_id" in k or "id" in k for k in found_keys)

    def test_model_card_markdown_contains_hugiml(self, fitted_clf_synthetic):
        card = self._basic_card(fitted_clf_synthetic)
        md = card.to_markdown()
        assert "HUGIML" in md or "HUG" in md or "hug" in md.lower()


# ---------------------------------------------------------------------------
# GovernanceMetadata
# ---------------------------------------------------------------------------


class TestGovernanceMetadata:
    def test_instantiate(self):
        meta = GovernanceMetadata(
            model_id="test-001",
            owner="test-owner",
            purpose="unit-test",
        )
        assert meta is not None

    def test_to_dict(self):
        meta = GovernanceMetadata(
            model_id="test-001",
            owner="test-owner",
            purpose="credit-scoring",
        )
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert "model_id" in d

    def test_to_json_round_trip(self):
        meta = GovernanceMetadata(
            model_id="test-002",
            owner="ci-runner",
            purpose="testing",
        )
        j = meta.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)
        assert parsed["model_id"] == "test-002"

    def test_default_review_status(self):
        meta = GovernanceMetadata(model_id="test-003")
        assert meta.review_status == "draft"

    def test_tags_default_empty(self):
        meta = GovernanceMetadata(model_id="test-004")
        assert isinstance(meta.tags, list)


# ---------------------------------------------------------------------------
# AuditArtifact
# ---------------------------------------------------------------------------


class TestAuditArtifact:
    def test_instantiate(self):
        artifact = AuditArtifact(model_id="audit-001")
        assert artifact is not None

    def test_has_model_id(self):
        artifact = AuditArtifact(model_id="audit-002")
        assert artifact.model_id == "audit-002"

    def test_artifact_to_dict(self):
        artifact = AuditArtifact(model_id="audit-003")
        d = artifact.to_dict()
        assert isinstance(d, dict)
        assert "model_id" in d

    def test_artifact_to_json(self):
        artifact = AuditArtifact(model_id="audit-004")
        j = artifact.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_artifact_save(self, tmp_path):
        artifact = AuditArtifact(model_id="audit-005")
        out = tmp_path / "audit.json"
        artifact.save(str(out))
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["model_id"] == "audit-005"


# ---------------------------------------------------------------------------
# package_audit_artifacts
# ---------------------------------------------------------------------------


class TestPackageAuditArtifacts:
    def test_package_creates_model_card(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        package_audit_artifacts(clf, "pkg-test-001", str(tmp_path))
        # At minimum, the model card JSON should be created
        assert (tmp_path / "model_card.json").exists()

    def test_package_creates_manifest(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        manifest_path = package_audit_artifacts(clf, "pkg-test-002", str(tmp_path))
        assert Path(manifest_path).exists()

    def test_package_manifest_is_json(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        manifest_path = package_audit_artifacts(clf, "pkg-test-003", str(tmp_path))
        data = json.loads(Path(manifest_path).read_text())
        assert isinstance(data, dict)

    def test_package_with_governance(self, fitted_clf_synthetic, tmp_path):
        clf, _, _ = fitted_clf_synthetic
        gov = GovernanceMetadata(
            model_id="pkg-gov-001",
            owner="test-owner",
            purpose="unit test",
        )
        package_audit_artifacts(clf, "pkg-gov-001", str(tmp_path), governance=gov)
        assert (tmp_path / "model_card.json").exists()
