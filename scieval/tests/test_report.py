import pandas as pd

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
