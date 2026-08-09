"""Packaging metadata checks for optional dependencies and source archives."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_includes_experiment_runners_and_excludes_generated_results() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include experiments" in manifest
    assert "*.py" in manifest
    assert "*.md" in manifest
    assert "prune experiments/benchmark/results" in manifest
    assert "prune experiments/scalability/results" in manifest


def test_optional_dependency_groups_cover_documented_install_surfaces() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dashboard_block = pyproject.split("dashboard = [", 1)[1].split("]", 1)[0]
    benchmarks_block = pyproject.split("benchmarks = [", 1)[1].split("]", 1)[0]
    all_block = pyproject.split("all = [", 1)[1].split("]", 1)[0]

    assert '"matplotlib>=3.5"' in dashboard_block
    assert '"statsmodels>=0.14"' in benchmarks_block
    assert "dashboard" in all_block
    assert "benchmarks" in all_block


def test_release_workflow_location_and_source_archive_checks() -> None:
    workflow_path = ROOT / ".github" / "workflows" / "release.yml"
    assert workflow_path.is_file()
    assert not (ROOT / "release.yml").exists()

    workflow = workflow_path.read_text(encoding="utf-8")
    assert "/experiments/benchmark/benchmark_dashboard\\.py$" in workflow
    assert "/experiments/benchmark/benchmarkREADME\\.md$" in workflow
    assert "/experiments/scalability/scalability_dashboard\\.py$" in workflow
    assert "/experiments/scalability/scalabilityREADME\\.md$" in workflow
    assert "merge_hugiml_results" not in workflow


def test_container_scan_is_retained_but_does_not_gate_python_publication() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    scan_block = workflow.split("  container-scan:", 1)[1].split(
        "\n  publish:", 1
    )[0]
    publish_block = workflow.split("  publish:", 1)[1]

    assert "run_container_scan:" in workflow
    assert "continue-on-error: true" in scan_block
    assert "actions/download-artifact@v8" in scan_block
    assert "wheels-ubuntu-latest-x86_64" in scan_block
    assert "docker/Dockerfile.scan" in scan_block
    assert 'exit-code: "0"' in scan_block
    assert "hashFiles('trivy-results.sarif') != ''" in scan_block
    assert "needs: [build-wheels, build-sdist, sbom]" in publish_block
    assert "container-scan" not in publish_block.split("steps:", 1)[0]


def test_editable_native_build_uses_development_optimization() -> None:
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert 'arg in {"editable_wheel", "develop"}' in setup_source
    assert "or _is_editable_build" in setup_source


def test_binding_compile_flags_preserve_pybind11_language_level() -> None:
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "def _binding_compile_args" in setup_source
    assert "_project_algo_args = tuple(algo_args)" in setup_source
    assert "inherited = [arg for arg in extra_postargs if arg not in _project_algo_args]" in setup_source
    assert "return [*bind_args, *inherited]" in setup_source
    assert "extra_postargs = _binding_compile_args(extra_postargs)" in setup_source
    assert "extra_postargs = bind_args" not in setup_source


def test_macos_ci_uses_a_serial_native_editable_build() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert 'echo "CC=/usr/bin/clang"' in workflow
    assert 'echo "CXX=/usr/bin/clang++"' in workflow
    assert 'echo "ARCHFLAGS=-arch $(uname -m)"' in workflow
    assert 'echo "HUGIML_NO_BREW_LIBOMP=1"' in workflow
    assert 'echo "HUGIML_FAST_BUILD=1"' in workflow
    assert 'echo "HUGIML_BUILD_BATCH_SIZE=1"' in workflow
    assert 'echo "HUGIML_BUILD_JOBS=1"' in workflow
