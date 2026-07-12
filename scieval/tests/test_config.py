from pathlib import Path

import pytest

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
