"""Keep optional presentation libraries outside Dash and compute startup."""

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    any(find_spec(name) is None for name in ("dash", "dash_bootstrap_components")),
    reason="Optional dashboard dependencies are not installed",
)


@pytest.mark.parametrize("target", ["dash", "worker"])
def test_dashboard_import_boundary(target):
    source = Path(__file__).resolve().parents[2] / "src"
    code = """
import sys
sys.path.insert(0, sys.argv[1])
if sys.argv[2] == 'dash':
    from hugiml.dashboard.dash_app import create_app
    app = create_app(False)
    with app.server.test_client() as client:
        for endpoint in ('/', '/_dash-layout', '/_dash-dependencies'):
            assert client.get(endpoint).status_code == 200
else:
    from hugiml.dashboard.dash_components.model_modes import run_dash_experiments
    from hugiml.dashboard.dash_components.data_utils import load_demo
    data = load_demo('credit')
    runs = run_dash_experiments(
        {'Interpretable Models': ['HUGIML']},
        {'HUGIML': [{'L': 1, 'topK': 10, 'n_jobs': 1}]},
        data['X'], data['y'], 3, 2026)
    assert runs[0]['status'] == 'ok', runs[0].get('diagnostic')
for package in ('streamlit', 'matplotlib'):
    assert not any(name == package or name.startswith(package + '.')
                   for name in sys.modules), package
"""
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", code, str(source), target],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
