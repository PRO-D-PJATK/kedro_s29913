"""Integration test: run the full pipeline in the `test` environment."""
from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_kedro_run_test_env():
    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(project_path=PROJECT_ROOT, env="test") as session:
        session.run()
    assert (PROJECT_ROOT / "data/test/08_reporting/metrics.json").exists()
