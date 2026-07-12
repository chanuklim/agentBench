from pathlib import Path

import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

from scieval.tasks.spec import BenchmarkSpec

# 벤치마크별 headline 지표 선택 우선순위 (없으면 첫 metric 사용)
_METRIC_PRIORITY = ["accuracy", "mean"]


def _headline(metrics: dict) -> tuple[str, float, float | None]:
    stderr = metrics["stderr"].value if "stderr" in metrics else None
    for name in _METRIC_PRIORITY:
        if name in metrics:
            return name, metrics[name].value, stderr
    name = next(iter(metrics))
    return name, metrics[name].value, stderr


def collect_results(run_dir: Path, catalog: dict[str, BenchmarkSpec]) -> pd.DataFrame:
    by_task_name = {s.task_name: s for s in catalog.values()}
    rows = []
    for info in list_eval_logs(str(run_dir / "logs")):
        log = read_eval_log(info, header_only=True)
        task_name = log.eval.task.split("/")[-1]
        spec = by_task_name.get(task_name)
        usage = log.stats.model_usage if log.stats else {}
        in_tok = sum(u.input_tokens for u in usage.values())
        out_tok = sum(u.output_tokens for u in usage.values())
        metric_name, metric_value, stderr = ("none", float("nan"), None)
        total = completed = 0
        if log.results and log.results.scores:
            metric_name, metric_value, stderr = _headline(log.results.scores[0].metrics)
            total = log.results.total_samples
            completed = log.results.completed_samples
        rows.append({
            "run_id": run_dir.name,
            "benchmark_id": spec.id if spec else task_name,
            "axis": spec.axis if spec else "unknown",
            "task_name": task_name,
            "model": log.eval.model,
            "status": log.status,
            "headline_metric": metric_name,
            "headline_value": metric_value,
            "stderr": stderr,
            "total_samples": total,
            "completed_samples": completed,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            # 스펙 §9: 완료율 <95%면 run 무효 — 리더보드 편입 금지
            "valid": total > 0 and completed / total >= 0.95,
        })
    return pd.DataFrame(rows)


def write_results(df: pd.DataFrame, run_dir: Path) -> Path:
    p = run_dir / "results.parquet"
    df.to_parquet(p, index=False)
    return p
