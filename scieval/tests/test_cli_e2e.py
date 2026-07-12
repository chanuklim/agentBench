import json
from pathlib import Path

from typer.testing import CliRunner

from scieval.runner.cli import app

FIXTURE = Path(__file__).parent / "fixtures" / "config"
runner = CliRunner()


def test_run_then_score(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("SCIEVAL_HOME", str(tmp_path))
    monkeypatch.setenv("SCIEVAL_TEST_CATALOG", "1")  # toy 카탈로그 주입

    r = runner.invoke(app, ["run", "--suite", "toy", "--model", "mock",
                            "--budget", "small", "--profile", "local",
                            "--config-dir", str(FIXTURE)])
    assert r.exit_code == 0, r.output
    run_dir = next((tmp_path / "runs").iterdir())
    assert (run_dir / "manifest.json").exists()

    r2 = runner.invoke(app, ["score", str(run_dir), "--config-dir", str(FIXTURE)])
    assert r2.exit_code == 0, r2.output
    assert (run_dir / "results.parquet").exists()
    scores = json.loads((run_dir / "scores.json").read_text())
    assert "mockllm/model" in scores
    assert (run_dir / "report.html").exists()
