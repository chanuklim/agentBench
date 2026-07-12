import json
from pathlib import Path

import pytest

from scieval.config import load_config
from scieval.runner.orchestrate import EnvError, ProfileError, run_suite
from scieval.tasks.spec import BenchmarkSpec

FIXTURE = Path(__file__).parent / "fixtures" / "config"


def _toy_catalog() -> dict[str, BenchmarkSpec]:
    def loader(**kw):
        from tests.fixtures.toy import toy

        return toy()

    return {
        "toy": BenchmarkSpec(
            id="toy", axis="reasoning", task_name="toy", agentic=False, gated=False,
            requires=set(), default_epochs=1, judge_mode="none",
            dataset_revision=None, loader=loader,
        )
    }


@pytest.fixture()
def cfg(monkeypatch):
    monkeypatch.setenv("TEST_BASE_URL", "http://localhost:8000/v1")
    return load_config(FIXTURE)


def test_run_suite_mockllm(cfg, tmp_path):
    out = run_suite(
        cfg, FIXTURE, suite="toy", model_ids=["mock"], budget="small",
        profile="local", catalog=_toy_catalog(), home=tmp_path,
    )
    assert out.success
    # manifest.json is written right after new_run_dir(), before eval_set()
    # runs -- it exists whenever run_dir exists, independent of eval outcome.
    m = json.loads((out.run_dir / "manifest.json").read_text())
    assert m["suite"] == "toy" and m["budget"] == "small"
    assert list(out.log_dir.glob("*.eval")), "eval log written"


def test_manifest_written_before_eval_runs(cfg, tmp_path, monkeypatch):
    """build_manifest/write_manifest happen right after new_run_dir(home),
    so manifest.json exists as soon as run_dir exists -- even if eval_set()
    itself blows up. All inputs to the manifest are known pre-eval."""
    def boom(**kw):
        raise RuntimeError("eval_set exploded")

    monkeypatch.setattr("scieval.runner.orchestrate.eval_set", boom)

    run_dirs_before = set((tmp_path / "runs").glob("*")) if (tmp_path / "runs").exists() else set()
    with pytest.raises(RuntimeError, match="eval_set exploded"):
        run_suite(cfg, FIXTURE, "toy", ["mock"], "small", "local", _toy_catalog(), tmp_path)
    run_dirs_after = set((tmp_path / "runs").glob("*"))
    new_dirs = run_dirs_after - run_dirs_before
    assert len(new_dirs) == 1
    assert (new_dirs.pop() / "manifest.json").exists()


def test_profile_requires_rejected(cfg, tmp_path):
    cat = _toy_catalog()
    cat["toy"] = BenchmarkSpec(**{**cat["toy"].__dict__, "requires": {"linux"}})
    with pytest.raises(ProfileError, match="linux"):
        run_suite(cfg, FIXTURE, "toy", ["mock"], "small", "local", cat, tmp_path)


def test_missing_env_rejected(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_BASE_URL", "http://localhost:8000/v1")
    cfg2 = cfg.model_copy(deep=True)
    cfg2.models["toy-remote"].base_url_env = "NOT_SET_XYZ"
    with pytest.raises(EnvError, match="NOT_SET_XYZ"):
        run_suite(cfg2, FIXTURE, "toy", ["toy-remote"], "small", "local",
                  _toy_catalog(), tmp_path)


def test_generate_args_wired_into_eval_set(cfg, tmp_path, monkeypatch):
    """generate_args(model) is forwarded into eval_set() as extra kwargs
    (e.g. max_tokens) when a single model is run."""
    captured: dict = {}

    def fake_eval_set(**kwargs):
        captured.update(kwargs)
        return True, []

    monkeypatch.setattr("scieval.runner.orchestrate.eval_set", fake_eval_set)

    cfg2 = cfg.model_copy(deep=True)
    cfg2.models["mock"].max_tokens = 1234

    out = run_suite(cfg2, FIXTURE, "toy", ["mock"], "small", "local",
                     _toy_catalog(), tmp_path)
    assert out.success
    assert captured["max_tokens"] == 1234


def test_generate_args_conflict_raises(cfg, tmp_path, monkeypatch):
    """When models in one run carry different generate args, eval_set()
    would silently collapse them onto a single GenerateConfig -- reject
    the run instead of running with a mismatch."""
    monkeypatch.setattr(
        "scieval.runner.orchestrate.eval_set", lambda **kw: (True, [])
    )

    cfg2 = cfg.model_copy(deep=True)
    cfg2.models["mock"].max_tokens = 1234
    cfg2.models["toy-remote"].max_tokens = 500

    with pytest.raises(ValueError, match="generate args"):
        run_suite(cfg2, FIXTURE, "toy", ["mock", "toy-remote"], "small", "local",
                  _toy_catalog(), tmp_path)


def test_allowed_suites_restriction_raises(cfg, tmp_path):
    cfg2 = cfg.model_copy(deep=True)
    cfg2.profiles["local"].allowed_suites = ["other_suite"]
    with pytest.raises(ProfileError, match="not allowed"):
        run_suite(cfg2, FIXTURE, "toy", ["mock"], "small", "local",
                  _toy_catalog(), tmp_path)


def test_allowed_suites_force_overrides_restriction(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scieval.runner.orchestrate.eval_set", lambda **kw: (True, [])
    )
    cfg2 = cfg.model_copy(deep=True)
    cfg2.profiles["local"].allowed_suites = ["other_suite"]
    out = run_suite(cfg2, FIXTURE, "toy", ["mock"], "small", "local",
                     _toy_catalog(), tmp_path, force=True)
    assert out.success


def test_allowed_suites_none_means_unrestricted(cfg, tmp_path, monkeypatch):
    """allowed_suites=None (the fixture default) never raises, force or not."""
    monkeypatch.setattr(
        "scieval.runner.orchestrate.eval_set", lambda **kw: (True, [])
    )
    assert cfg.profiles["local"].allowed_suites is None
    out = run_suite(cfg, FIXTURE, "toy", ["mock"], "small", "local",
                     _toy_catalog(), tmp_path)
    assert out.success
