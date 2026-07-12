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
    m = json.loads((out.run_dir / "manifest.json").read_text())
    assert m["suite"] == "toy" and m["budget"] == "small"
    assert list(out.log_dir.glob("*.eval")), "eval log written"


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
