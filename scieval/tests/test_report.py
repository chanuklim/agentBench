import pandas as pd
import pytest

from scieval.reporting.report import ModelReport, render_report
from scieval.scoring.aggregate import AxisScore, IndexResult


def test_render_report(tmp_path):
    (tmp_path / "manifest.json").write_text('{"run_id": "r1", "suite": "toy"}')
    df = pd.DataFrame([{
        "run_id": "r1", "benchmark_id": "gpqa_diamond", "axis": "reasoning",
        "model": "mockllm/model", "status": "success", "headline_metric": "accuracy",
        "headline_value": 0.61, "stderr": 0.03, "total_samples": 198,
        "completed_samples": 198, "input_tokens": 100, "output_tokens": 50,
        "total_tokens": 150, "task_name": "gpqa_diamond",
    }])
    per_model = {
        "mockllm/model": ModelReport(
            axes={"reasoning": AxisScore(score=0.62, coverage=0.86)},
            index=IndexResult(value=0.62, partial=True),
            gates={"reasoning": "pass"},
            normalized={"gpqa_diamond": 0.64},
        )
    }
    p = render_report(tmp_path, df, per_model)
    html = p.read_text()
    assert "gpqa_diamond" in html and "0.61" in html
    assert "PARTIAL" in html          # 결측 축 경고
    assert "coverage" in html.lower()
    assert "r1" in html


def test_render_report_with_nan_headline(tmp_path):
    (tmp_path / "manifest.json").write_text('{"run_id": "r2", "suite": "toy"}')
    df = pd.DataFrame([{
        "run_id": "r2", "benchmark_id": "test_bench", "axis": "reasoning",
        "model": "mockllm/model", "status": "error", "headline_metric": "accuracy",
        "headline_value": float("nan"), "stderr": None, "total_samples": 100,
        "completed_samples": 0, "input_tokens": 0, "output_tokens": 0,
        "total_tokens": 0, "task_name": "test_bench",
    }])
    per_model = {
        "mockllm/model": ModelReport(
            axes={"reasoning": AxisScore(score=0.0, coverage=0.0)},
            index=IndexResult(value=0.0, partial=True),
            gates={"reasoning": "fail"},
            normalized={"test_bench": 0.0},
        )
    }
    p = render_report(tmp_path, df, per_model)
    html = p.read_text(encoding="utf-8")
    assert "test_bench" in html
    assert "error" in html
    assert "nan" not in html
    assert "-" in html


def test_render_report_with_nan_stderr(tmp_path):
    """stderr=NaN (as opposed to None) must also render as '-', not 'nan' --
    the headline cell already had this guard, the stderr cell didn't."""
    (tmp_path / "manifest.json").write_text('{"run_id": "r3", "suite": "toy"}')
    df = pd.DataFrame([{
        "run_id": "r3", "benchmark_id": "std_err_test_bench", "axis": "reasoning",
        "model": "mockllm/model", "status": "success", "headline_metric": "accuracy",
        "headline_value": 0.5, "stderr": float("nan"), "total_samples": 10,
        "completed_samples": 10, "input_tokens": 0, "output_tokens": 0,
        "total_tokens": 0, "task_name": "std_err_test_bench",
    }])
    per_model = {
        "mockllm/model": ModelReport(
            axes={"reasoning": AxisScore(score=0.5, coverage=1.0)},
            index=IndexResult(value=0.5, partial=False),
            gates={"reasoning": "pass"},
            normalized={"std_err_test_bench": 0.5},
        )
    }
    p = render_report(tmp_path, df, per_model)
    html = p.read_text(encoding="utf-8")
    assert "std_err_test_bench" in html
    assert "nan" not in html.lower()


def test_render_report_missing_manifest_raises_clear_error(tmp_path):
    """render_report on a directory with no manifest.json should fail with
    a clear, actionable message rather than a bare FileNotFoundError from
    deep inside Path.read_text()."""
    df = pd.DataFrame([])
    with pytest.raises(FileNotFoundError, match="manifest.json not found"):
        render_report(tmp_path, df, {})
