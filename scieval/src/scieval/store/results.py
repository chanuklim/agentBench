from pathlib import Path

import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import ModelUsage

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


def _split_usage(
    usage: dict[str, ModelUsage], eval_model: str
) -> tuple[list[ModelUsage], list[ModelUsage]]:
    """Split per-model token usage into (own, other).

    `usage` (from ``log.stats.model_usage``) is keyed by model name and
    mixes together every distinct model invoked during the eval — the
    evaluated model *and* any judge/grader models called via
    ``model_roles``. "own" is usage recorded under `eval_model`'s exact
    key; "other" is everything else (judge/grader usage).

    If no key matches `eval_model` exactly but `usage` is non-empty
    (e.g. a single-model log whose key format differs slightly from
    `log.eval.model`), treat every entry as "own" — we never want a
    normal single-model run to silently show zeroed token columns.
    """
    own = [u for k, u in usage.items() if k == eval_model]
    other = [u for k, u in usage.items() if k != eval_model]
    if not own and usage:
        return list(usage.values()), []
    return own, other


def _sum_tokens(usages: list[ModelUsage]) -> int:
    """Sum ModelUsage.total_tokens (incl. cache/reasoning tokens).

    ModelUsage.total_tokens is a required int (default 0) rather than
    input_tokens + output_tokens: input_tokens explicitly excludes
    cache read/write tokens, so total_tokens is inspect_ai's own,
    correct total. Falls back to input_tokens + output_tokens per
    entry when total_tokens is unset (0), since a provider that
    doesn't report a total shouldn't make the column silently zero.
    """
    total = 0
    for u in usages:
        total += u.total_tokens if u.total_tokens else (u.input_tokens + u.output_tokens)
    return total


def collect_results(run_dir: Path, catalog: dict[str, BenchmarkSpec]) -> pd.DataFrame:
    by_task_name = {s.task_name: s for s in catalog.values()}
    rows = []
    for info in list_eval_logs(str(run_dir / "logs")):
        log = read_eval_log(info, header_only=True)
        task_name = log.eval.task.split("/")[-1]
        spec = by_task_name.get(task_name)
        own_usage, judge_usage = _split_usage(log.stats.model_usage, log.eval.model)
        in_tok = sum(u.input_tokens for u in own_usage)
        out_tok = sum(u.output_tokens for u in own_usage)
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
            "total_tokens": _sum_tokens(own_usage),
            "judge_total_tokens": _sum_tokens(judge_usage),
            # 스펙 §9: 완료율 <95%면 run 무효 — 리더보드 편입 금지
            "valid": total > 0 and completed / total >= 0.95,
        })
    return pd.DataFrame(rows)


def write_results(df: pd.DataFrame, run_dir: Path) -> Path:
    p = run_dir / "results.parquet"
    df.to_parquet(p, index=False)
    return p
