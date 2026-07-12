from pathlib import Path

import pytest
from inspect_ai.model import ModelUsage

from scieval.config import load_config
from scieval.runner.orchestrate import run_suite
from scieval.store.results import _split_usage, _sum_tokens, collect_results, write_results

FIXTURE = Path(__file__).parent / "fixtures" / "config"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    import os

    os.environ["TEST_BASE_URL"] = "http://localhost:8000/v1"
    from tests.test_orchestrate import _toy_catalog

    cfg = load_config(FIXTURE)
    home = tmp_path_factory.mktemp("home")
    out = run_suite(cfg, FIXTURE, "toy", ["mock"], "small", "local",
                    _toy_catalog(), home)
    return out.run_dir


def test_collect_results(run_dir):
    from tests.test_orchestrate import _toy_catalog

    df = collect_results(run_dir, _toy_catalog())
    assert len(df) == 1
    row = df.iloc[0]
    assert row["benchmark_id"] == "toy"
    assert row["axis"] == "reasoning"
    assert row["model"].startswith("mockllm")
    assert row["total_samples"] == 2
    assert row["completed_samples"] == 2
    assert 0.0 <= row["headline_value"] <= 1.0
    assert row["total_tokens"] >= 0
    assert "judge_total_tokens" in df.columns
    assert row["judge_total_tokens"] == 0  # toy run에는 judge/grader 모델이 없음
    assert bool(row["valid"]) is True  # 완료율 ≥95% (스펙 §9)


def test_write_results(run_dir):
    from tests.fixtures import toy  # noqa: F401
    from tests.test_orchestrate import _toy_catalog
    import pandas as pd

    df = collect_results(run_dir, _toy_catalog())
    p = write_results(df, run_dir)
    assert p == run_dir / "results.parquet"
    assert len(pd.read_parquet(p)) == 1


# --- _split_usage / _sum_tokens: unit tests with synthetic ModelUsage ---

def test_split_usage_separates_own_from_judge():
    usage = {
        "openai/gpt-4o": ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        "openai/gpt-4o-mini": ModelUsage(input_tokens=100, output_tokens=50, total_tokens=150),
    }
    own, other = _split_usage(usage, "openai/gpt-4o")
    assert own == [usage["openai/gpt-4o"]]
    assert other == [usage["openai/gpt-4o-mini"]]


def test_split_usage_no_key_match_falls_back_to_all_own():
    # single-model log whose model_usage key format doesn't exactly
    # match log.eval.model — should never silently zero the columns
    usage = {"mockllm/model": ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)}
    own, other = _split_usage(usage, "mockllm/model:different-suffix")
    assert own == [usage["mockllm/model"]]
    assert other == []


def test_split_usage_empty_dict():
    own, other = _split_usage({}, "mockllm/model")
    assert own == []
    assert other == []


def test_sum_tokens_uses_total_tokens_field():
    # total_tokens (15) intentionally differs from input+output (10) to
    # prove cache/reasoning-inclusive total_tokens is used, not the sum
    usages = [ModelUsage(input_tokens=5, output_tokens=5, total_tokens=15)]
    assert _sum_tokens(usages) == 15


def test_sum_tokens_falls_back_when_total_tokens_unset():
    # total_tokens defaults to 0 when a provider doesn't report it
    usages = [ModelUsage(input_tokens=7, output_tokens=3, total_tokens=0)]
    assert _sum_tokens(usages) == 10


def test_sum_tokens_empty_list_is_zero():
    assert _sum_tokens([]) == 0
