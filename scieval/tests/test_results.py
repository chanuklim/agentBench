from pathlib import Path

import pytest

from scieval.config import load_config
from scieval.runner.orchestrate import run_suite
from scieval.store.results import collect_results, write_results

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
    assert bool(row["valid"]) is True  # 완료율 ≥95% (스펙 §9)


def test_write_results(run_dir):
    from tests.fixtures import toy  # noqa: F401
    from tests.test_orchestrate import _toy_catalog
    import pandas as pd

    df = collect_results(run_dir, _toy_catalog())
    p = write_results(df, run_dir)
    assert p == run_dir / "results.parquet"
    assert len(pd.read_parquet(p)) == 1
