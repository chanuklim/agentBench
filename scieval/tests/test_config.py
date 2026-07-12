import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from scieval.config import ConfigError, load_config

FIXTURE = Path(__file__).parent / "fixtures" / "config"


def test_load_config(monkeypatch):
    monkeypatch.setenv("TEST_BASE_URL", "http://localhost:8000/v1")
    cfg = load_config(FIXTURE)
    assert cfg.budgets["small"].token_limit == "200k"
    assert cfg.budgets["small"].message_limit == 30
    m = cfg.models["toy-remote"]
    assert m.inspect_model == "openai-api/toy/toy-model"
    assert m.pricing.input_per_mtok == 0.5
    assert cfg.profiles["local"].provides == {"docker", "network"}
    assert cfg.suites["toy"].entries[0].benchmark == "toy"
    assert cfg.weights.axes["reasoning"].weight == 22
    assert cfg.anchors["gpqa_diamond"].naive == 0.25
    assert cfg.judges["hle_stem"] == "openai/gpt-4o-mini"


def test_env_expansion_missing(monkeypatch):
    monkeypatch.delenv("TEST_BASE_URL", raising=False)
    with pytest.raises(ConfigError, match="TEST_BASE_URL"):
        load_config(FIXTURE)


def test_unknown_key_rejected(tmp_path, monkeypatch):
    """extra='forbid' on config models catches YAML key typos at load time
    instead of silently dropping them (final-review fix wave, item 1)."""
    monkeypatch.setenv("TEST_BASE_URL", "http://localhost:8000/v1")
    copied_dir = tmp_path / "config"
    shutil.copytree(FIXTURE, copied_dir)

    budgets_path = copied_dir / "budgets.yaml"
    budgets = yaml.safe_load(budgets_path.read_text())
    budgets["small"]["cost_limt"] = 2.0  # typo: should be cost_limit
    budgets_path.write_text(yaml.dump(budgets))

    with pytest.raises(ValidationError, match="cost_limt"):
        load_config(copied_dir)


def test_unknown_key_rejected_nested_axis_weights(tmp_path, monkeypatch):
    """Same guard for a nested model (AxisWeights inside scoring.yaml)."""
    monkeypatch.setenv("TEST_BASE_URL", "http://localhost:8000/v1")
    copied_dir = tmp_path / "config"
    shutil.copytree(FIXTURE, copied_dir)

    scoring_path = copied_dir / "scoring.yaml"
    scoring = yaml.safe_load(scoring_path.read_text())
    scoring["weights"]["axes"]["reasoning"]["wieght"] = 22  # typo: should be weight
    scoring_path.write_text(yaml.dump(scoring))

    with pytest.raises(ValidationError, match="wieght"):
        load_config(copied_dir)
