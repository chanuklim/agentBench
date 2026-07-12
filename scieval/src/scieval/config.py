import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ConfigError(Exception):
    pass


class BudgetClass(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_limit: str
    message_limit: int
    working_limit_s: int
    cost_limit: float | None = None


class Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = ""
    inspect_model: str
    base_url_env: str | None = None
    api_key_env: str | None = None
    pricing: Pricing | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    seed: int | None = None
    note: str | None = None


class ProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    provides: set[str] = Field(default_factory=set)
    allowed_suites: list[str] | None = None


class SuiteEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: str
    limit: int | None = None
    epochs: int | None = None


class SuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    entries: list[SuiteEntry]


class Anchor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    naive: float
    expert: float | None = None
    quality: Literal["sourced", "provisional", "fallback"] = "provisional"


class AxisWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight: float
    benchmarks: dict[str, float]


class WeightsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axes: dict[str, AxisWeights]


class ScievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelConfig]
    budgets: dict[str, BudgetClass]
    profiles: dict[str, ProfileConfig]
    suites: dict[str, SuiteConfig]
    anchors: dict[str, Anchor]
    weights: WeightsConfig
    judges: dict[str, str]


_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(node):
    if isinstance(node, str):
        def sub(m: re.Match) -> str:
            val = os.environ.get(m.group(1))
            if val is None:
                raise ConfigError(f"environment variable not set: {m.group(1)}")
            return val
        return _ENV_RE.sub(sub, node)
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    return node


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"missing config file: {path.name}")
    return _expand_env(yaml.safe_load(path.read_text()) or {})


def load_config(config_dir: Path) -> ScievalConfig:
    budgets = {k: BudgetClass(**v) for k, v in _load_yaml(config_dir / "budgets.yaml").items()}
    models = {k: ModelConfig(id=k, **v) for k, v in _load_yaml(config_dir / "models.yaml").items()}
    profiles = {k: ProfileConfig(name=k, **v) for k, v in _load_yaml(config_dir / "profiles.yaml").items()}
    suites = {k: SuiteConfig(name=k, **v) for k, v in _load_yaml(config_dir / "suites.yaml").items()}
    scoring = _load_yaml(config_dir / "scoring.yaml")
    return ScievalConfig(
        models=models,
        budgets=budgets,
        profiles=profiles,
        suites=suites,
        anchors={k: Anchor(**v) for k, v in scoring.get("anchors", {}).items()},
        weights=WeightsConfig(**scoring.get("weights", {"axes": {}})),
        judges=scoring.get("judges", {}),
    )
