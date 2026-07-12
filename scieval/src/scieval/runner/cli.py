import json
import os
from pathlib import Path

import typer

from scieval.config import load_config
from scieval.paths import scieval_home
from scieval.providers.health import check_endpoint
from scieval.runner.orchestrate import run_suite
from scieval.reporting.report import ModelReport, render_report
from scieval.scoring.aggregate import axis_scores, final_index
from scieval.scoring.gates import evaluate as evaluate_gates
from scieval.scoring.normalize import normalize
from scieval.store.results import collect_results, write_results
from scieval.tasks.catalog import CATALOG
from scieval.tasks.spec import BenchmarkSpec

app = typer.Typer(no_args_is_help=True)

# 커밋 상용 judge 모델의 provider prefix -> 표준 API 키 env var.
# 알려지지 않은 prefix(예: 사내 커스텀 judge)는 검사를 건너뛴다 — 계정별
# env var 관례를 우리가 알 수 없기 때문.
_JUDGE_PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def _missing_judge_env_vars(
    judge_models: list[str], env: dict[str, str] | None = None
) -> list[str]:
    """judge 모델 문자열(예: "openai/gpt-5-mini")을 provider prefix로 매핑해
    필요한 API 키 env var 중 설정되지 않은 것의 이름 목록을 반환한다.

    Task 6 review 후속: run의 preflight는 실행 대상 모델의 env만 검사했고
    judge 모델(cfg.judges의 값, raw inspect model 문자열)은 env 메타데이터가
    없어 검사되지 않았다. 알려진 prefix만 검사하고, 모르는 prefix는 건너뛴다.
    """
    env = os.environ if env is None else env
    missing: list[str] = []
    for jm in judge_models:
        prefix = jm.split("/", 1)[0]
        var = _JUDGE_PROVIDER_ENV.get(prefix)
        if var and not env.get(var) and var not in missing:
            missing.append(var)
    return missing


def _catalog() -> dict[str, BenchmarkSpec]:
    if os.environ.get("SCIEVAL_TEST_CATALOG") == "1":
        return _test_catalog()
    return CATALOG


def _test_catalog() -> dict[str, BenchmarkSpec]:
    def loader(**kw):
        from inspect_ai import Task
        from inspect_ai.dataset import Sample
        from inspect_ai.scorer import includes
        from inspect_ai.solver import generate

        return Task(
            dataset=[Sample(input="Say the word apple.", target="apple"),
                     Sample(input="Say the word banana.", target="banana")],
            solver=[generate()], scorer=includes(), name="toy",
        )

    return {"toy": BenchmarkSpec(
        id="toy", axis="reasoning", task_name="toy", agentic=False, gated=False,
        requires=set(), default_epochs=1, judge_mode="none",
        dataset_revision=None, loader=loader,
    )}


@app.command()
def run(
    suite: str = typer.Option(...),
    model: list[str] = typer.Option(...),
    budget: str = typer.Option("standard"),
    profile: str = typer.Option(...),
    config_dir: Path = typer.Option(Path("configs")),
    limit: int | None = typer.Option(None),
    skip_preflight: bool = typer.Option(False),
):
    cfg = load_config(config_dir)
    # 스펙 §9 preflight: 원격 endpoint 모델은 health-check 통과 후에만 실행
    if not skip_preflight:
        for mid in model:
            m = cfg.models[mid]
            if m.base_url_env and os.environ.get(m.base_url_env):
                r = check_endpoint(
                    os.environ[m.base_url_env],
                    os.environ.get(m.api_key_env or "") if m.api_key_env else None,
                    m.inspect_model.split("/")[-1],
                )
                if not r.ok:
                    typer.echo(f"preflight failed for {mid}: {r.detail}", err=True)
                    raise typer.Exit(1)
        judge_models = sorted({
            cfg.judges[e.benchmark]
            for e in cfg.suites[suite].entries
            if e.benchmark in cfg.judges
        })
        missing_judge_env = _missing_judge_env_vars(judge_models)
        if missing_judge_env:
            typer.echo(
                f"preflight failed: missing judge env vars: {missing_judge_env}",
                err=True,
            )
            raise typer.Exit(1)
    out = run_suite(cfg, config_dir, suite, list(model), budget, profile,
                    _catalog(), scieval_home(), limit=limit)
    typer.echo(f"run_dir: {out.run_dir}  success: {out.success}")
    raise typer.Exit(0 if out.success else 1)


@app.command()
def score(run_dir: Path, config_dir: Path = typer.Option(Path("configs"))):
    cfg = load_config(config_dir)
    catalog = _catalog()
    df = collect_results(run_dir, catalog)
    write_results(df, run_dir)
    per_model: dict[str, ModelReport] = {}
    gate_thresholds = {axis: None for axis in cfg.weights.axes}
    for model_name, g in df.groupby("model"):
        normalized = {
            row.benchmark_id: normalize(row.headline_value, cfg.anchors[row.benchmark_id])
            for row in g.itertuples()
            if row.benchmark_id in cfg.anchors and row.status == "success" and row.valid
        }
        axes = axis_scores(normalized, cfg.weights)
        per_model[str(model_name)] = ModelReport(
            axes=axes,
            index=final_index(axes, cfg.weights),
            gates=evaluate_gates(axes, gate_thresholds),
            normalized=normalized,
        )
    (run_dir / "scores.json").write_text(json.dumps({
        m: {
            "index": r.index.value, "partial": r.index.partial,
            "axes": {a: {"score": s.score, "coverage": s.coverage}
                     for a, s in r.axes.items()},
            "normalized": r.normalized, "gates": r.gates,
        } for m, r in per_model.items()
    }, indent=2))
    render_report(run_dir, df, per_model)
    typer.echo(f"scored {len(per_model)} model(s) → {run_dir / 'report.html'}")


@app.command()
def health(model_id: str, config_dir: Path = typer.Option(Path("configs"))):
    cfg = load_config(config_dir)
    m = cfg.models[model_id]
    base_url = os.environ.get(m.base_url_env or "", "")
    api_key = os.environ.get(m.api_key_env or "") if m.api_key_env else None
    if not base_url:
        typer.echo(f"env {m.base_url_env} not set", err=True)
        raise typer.Exit(1)
    r = check_endpoint(base_url, api_key, m.inspect_model.split("/")[-1])
    typer.echo(f"ok={r.ok} usage={r.usage_present} {r.detail}")
    raise typer.Exit(0 if r.ok else 1)


@app.command()
def fetch(suite: str, config_dir: Path = typer.Option(Path("configs"))):
    """suite의 벤치마크 데이터셋을 사전 다운로드 (gated는 HF_TOKEN 필요)."""
    cfg = load_config(config_dir)
    catalog = _catalog()
    for entry in cfg.suites[suite].entries:
        spec = catalog[entry.benchmark]
        typer.echo(f"fetching {spec.id} (gated={spec.gated}) ...")
        spec.loader()  # Task 생성이 데이터셋 다운로드를 트리거
    typer.echo("done")
