import shutil
from pathlib import Path

import yaml
from typer.testing import CliRunner

from scieval.runner.cli import _missing_judge_env_vars, app

FIXTURE = Path(__file__).parent / "fixtures" / "config"
runner = CliRunner()


def test_missing_judge_env_vars_reports_unset_known_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    missing = _missing_judge_env_vars(["openai/gpt-5-mini", "anthropic/claude-x"])
    assert missing == ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]


def test_missing_judge_env_vars_empty_when_all_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    missing = _missing_judge_env_vars(["openai/gpt-5-mini"])
    assert missing == []


def test_missing_judge_env_vars_skips_unknown_provider(monkeypatch):
    monkeypatch.delenv("SOME_OTHER_KEY", raising=False)
    missing = _missing_judge_env_vars(["mycompany/internal-judge"])
    assert missing == []


def test_missing_judge_env_vars_dedupes_same_var(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    missing = _missing_judge_env_vars(["openai/gpt-5-mini", "openai/gpt-5"])
    assert missing == ["OPENAI_API_KEY"]


def test_missing_judge_env_vars_no_judges():
    assert _missing_judge_env_vars([]) == []


def test_judge_preflight_wiring_missing_env_key_blocks_run(tmp_path, monkeypatch):
    """run() preflight checks judge env vars; missing OPENAI_API_KEY → exit 1."""
    # Copy fixture config to tmp_path
    copied_dir = tmp_path / "config"
    shutil.copytree(FIXTURE, copied_dir)

    # Modify scoring.yaml to add judges for toy benchmark (which is in toy suite)
    scoring_path = copied_dir / "scoring.yaml"
    with open(scoring_path) as f:
        scoring = yaml.safe_load(f)

    # Add judges key with toy judge requiring OPENAI_API_KEY
    if "judges" not in scoring:
        scoring["judges"] = {}
    scoring["judges"]["toy"] = "openai/gpt-5-mini"

    with open(scoring_path, "w") as f:
        yaml.dump(scoring, f)

    # Unset OPENAI_API_KEY and set test environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TEST_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("SCIEVAL_TEST_CATALOG", "1")
    monkeypatch.setenv("SCIEVAL_HOME", str(tmp_path))

    # Invoke run command without --skip-preflight
    result = runner.invoke(app, ["run", "--suite", "toy", "--model", "mock",
                                 "--budget", "small", "--profile", "local",
                                 "--config-dir", str(copied_dir)])

    # Should fail preflight due to missing OPENAI_API_KEY
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output
    assert "preflight failed" in result.output


def test_judge_preflight_skip_allows_run(tmp_path, monkeypatch):
    """run() with --skip-preflight bypasses judge env check."""
    # Copy fixture config to tmp_path
    copied_dir = tmp_path / "config"
    shutil.copytree(FIXTURE, copied_dir)

    # Modify scoring.yaml to add judges for toy benchmark (which is in toy suite)
    scoring_path = copied_dir / "scoring.yaml"
    with open(scoring_path) as f:
        scoring = yaml.safe_load(f)

    # Add judges key with toy judge requiring OPENAI_API_KEY
    if "judges" not in scoring:
        scoring["judges"] = {}
    scoring["judges"]["toy"] = "openai/gpt-5-mini"

    with open(scoring_path, "w") as f:
        yaml.dump(scoring, f)

    # Unset OPENAI_API_KEY and set test environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TEST_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("SCIEVAL_TEST_CATALOG", "1")
    monkeypatch.setenv("SCIEVAL_HOME", str(tmp_path))

    # Invoke run command WITH --skip-preflight
    result = runner.invoke(app, ["run", "--suite", "toy", "--model", "mock",
                                 "--budget", "small", "--profile", "local",
                                 "--config-dir", str(copied_dir), "--skip-preflight"])

    # Should succeed (exit 0) because preflight is skipped
    # toy benchmark doesn't need judge at eval time, so it will run successfully
    assert result.exit_code == 0


def test_score_empty_run_dir_exits_with_message(tmp_path, monkeypatch):
    """score on a run dir with no eval logs should fail cleanly (exit 1 +
    a clear message) instead of raising KeyError deep in the scoring code."""
    monkeypatch.setenv("TEST_BASE_URL", "http://localhost:8000/v1")
    run_dir = tmp_path / "empty-run"
    (run_dir / "logs").mkdir(parents=True)

    result = runner.invoke(app, ["score", str(run_dir), "--config-dir", str(FIXTURE)])

    assert result.exit_code == 1
    assert "no eval logs found" in result.output
    assert str(run_dir) in result.output
    assert not (run_dir / "results.parquet").exists()


def test_run_force_flag_overrides_allowed_suites(tmp_path, monkeypatch):
    """--force on `scieval run` threads through to run_suite(force=True) and
    bypasses an allowed_suites profile restriction; without it, the run is
    rejected before any eval happens."""
    copied_dir = tmp_path / "config"
    shutil.copytree(FIXTURE, copied_dir)

    profiles_path = copied_dir / "profiles.yaml"
    with open(profiles_path) as f:
        profiles = yaml.safe_load(f)
    profiles["local"]["allowed_suites"] = ["other_suite"]
    with open(profiles_path, "w") as f:
        yaml.dump(profiles, f)

    monkeypatch.setenv("TEST_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("SCIEVAL_TEST_CATALOG", "1")
    monkeypatch.setenv("SCIEVAL_HOME", str(tmp_path))

    blocked = runner.invoke(app, ["run", "--suite", "toy", "--model", "mock",
                                  "--budget", "small", "--profile", "local",
                                  "--config-dir", str(copied_dir), "--skip-preflight"])
    assert blocked.exit_code != 0

    forced = runner.invoke(app, ["run", "--suite", "toy", "--model", "mock",
                                 "--budget", "small", "--profile", "local",
                                 "--config-dir", str(copied_dir), "--skip-preflight",
                                 "--force"])
    assert forced.exit_code == 0, forced.output
