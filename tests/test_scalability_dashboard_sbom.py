# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Smoke tests for scalability dashboard reproducibility/SBOM assembly."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_scalability_dashboard():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments" / "scalability" / "scalability_dashboard.py"
    spec = importlib.util.spec_from_file_location("scalability_dashboard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scalability_dashboard_embeds_collapsed_privacy_sanitized_sbom(tmp_path) -> None:
    dash = _load_scalability_dashboard()
    outdir = tmp_path / "private-results-root"
    outdir.mkdir()
    checkpoint = outdir / "scalability_checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "metadata": {"generated_at": "2026-07-07T00:00:00+00:00", "sweeps": {}},
                "results": [
                    {
                        "key": "smoke",
                        "dataset": "sparse_nonlinear",
                        "section": "n_scaling",
                        "model": "hug_op_adaptive_sgd",
                        "n": 100,
                        "p": 20,
                        "status": "ok",
                        "fit_s": 0.1,
                        "predict_s": 0.01,
                        "auc": 0.75,
                        "patterns": 3,
                        "peak_process_tree_rss_mb": 128,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dash.build_outputs(argparse.Namespace(outdir=str(outdir), output_html=None, include_sbom=True))

    html = (outdir / "hugiml_scalability_dashboard.html").read_text(encoding="utf-8")
    sbom_text = (outdir / "scalability_reproducibility_sbom.json").read_text(encoding="utf-8")
    sbom = json.loads(sbom_text)

    assert 'id="sbomCard"' in html
    assert "<details><summary" in html
    assert "Reproducibility / SBOM manifest" in html
    assert "hug_op_adaptive_sgd" in html
    assert sbom["bom_format"] == "HUGIML scalability reproducibility manifest"
    assert str(outdir) not in sbom_text
    assert "<output-dir>" in sbom_text
