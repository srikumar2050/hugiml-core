from __future__ import annotations

from pathlib import Path

import pandas as pd

from hugiml.llm import ActionRequest, DatasetRegistry, HUGIMLActionOrchestrator
from hugiml.llm.planner import plan_request


def test_dataset_registry_lists_builtin_and_benchmark() -> None:
    repo = Path(__file__).resolve().parents[2]
    registry = DatasetRegistry(repo_root=repo)
    names = {item.name for item in registry.list_datasets(include_profiles=False)}
    assert "churn_synthetic" in names
    assert "BreastCancerOriginal" in names


def test_user_dataset_requires_and_preserves_explicit_target(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    registry = DatasetRegistry(repo_root=repo)
    source = tmp_path / "upload.csv"
    pd.DataFrame({"x": [0, 1, 2, 3, 4, 5], "z": [1, 1, 0, 0, 1, 0], "label": [0, 1, 0, 1, 0, 1]}).to_csv(source, index=False)
    info = registry.register_user_dataset(source, target_column="label", dataset_name="pytest_upload", overwrite=True)
    try:
        assert info.target == "label"
        X, y, loaded = registry.load_dataset("pytest_upload")
        assert loaded.target == "label"
        assert X.shape == (6, 2)
        assert list(y) == [0, 1, 0, 1, 0, 1]
    finally:
        path = Path(info.path or "")
        if path.exists():
            path.unlink()
        sidecar = path.with_suffix(path.suffix + ".target.json")
        if sidecar.exists():
            sidecar.unlink()


def test_refuses_code_edits_and_baseline_models() -> None:
    code = plan_request("rewrite the classifier source", prefer_llm=False)
    assert getattr(code, "refusal_reason", None) == "code_modification_not_supported"
    baseline = plan_request("train an xgboost baseline", prefer_llm=False)
    assert getattr(baseline, "refusal_reason", None) == "baseline_model_not_supported"


def test_hugiml_only_end_to_end_flow(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    orch = HUGIMLActionOrchestrator(repo_root=repo, session_dir=tmp_path)
    build = orch.execute(ActionRequest(action="build_model", dataset="churn_synthetic", strategy="fast"))
    assert build.ok
    assert "metrics" in build.tables
    explain = orch.execute(ActionRequest(action="explain_model", limit=3))
    assert explain.ok
    assert "metrics" in explain.tables
    prune = orch.execute(ActionRequest(action="prune_patterns", min_support=0.03, reason="pytest pruning"))
    assert prune.ok
    governance = orch.execute(ActionRequest(action="generate_governance_report"))
    assert governance.ok
    assert "model_card_md" in governance.artifacts
