from dataclasses import dataclass
from pathlib import Path

from inspect_ai import eval_set

from scieval.config import ScievalConfig
from scieval.paths import new_run_dir
from scieval.providers.registry import (
    ensure_model_info_registered,
    generate_args,
    model_cost_config,
    required_env,
)
from scieval.runner.manifest import build_manifest, write_manifest
from scieval.tasks.spec import BenchmarkSpec, prepare


class ProfileError(Exception):
    pass


class EnvError(Exception):
    pass


@dataclass(frozen=True)
class RunOutcome:
    run_dir: Path
    success: bool
    log_dir: Path


def run_suite(
    cfg: ScievalConfig,
    config_dir: Path,
    suite: str,
    model_ids: list[str],
    budget: str,
    profile: str,
    catalog: dict[str, BenchmarkSpec],
    home: Path,
    limit: int | None = None,
    force: bool = False,
) -> RunOutcome:
    prof = cfg.profiles[profile]
    if prof.allowed_suites is not None and suite not in prof.allowed_suites and not force:
        raise ProfileError(
            f"suite '{suite}' not allowed for profile '{profile}' "
            f"(allowed: {prof.allowed_suites}); use --force to override"
        )
    suite_cfg = cfg.suites[suite]
    budget_cfg = cfg.budgets[budget]
    models = {mid: cfg.models[mid] for mid in model_ids}

    specs = [catalog[e.benchmark] for e in suite_cfg.entries]
    missing_caps = set().union(*(s.requires for s in specs)) - prof.provides
    if missing_caps:
        raise ProfileError(
            f"profile '{profile}' lacks capabilities {sorted(missing_caps)} "
            f"required by suite '{suite}'"
        )
    for m in models.values():
        missing = required_env(m)
        if missing:
            raise EnvError(f"model '{m.id}' missing env vars: {missing}")

    run_dir = new_run_dir(home)
    manifest = build_manifest(
        run_id=run_dir.name, suite=suite, budget=budget, profile=profile,
        models=models, benchmarks=specs, judges=cfg.judges, config_dir=config_dir,
    )
    write_manifest(run_dir, manifest)

    log_dir = run_dir / "logs"
    cost_cfg = model_cost_config(models)
    # inspect_ai's set_model_cost() (called internally by eval_set() for each
    # model_cost_config entry) raises for models unknown to Inspect's model-info
    # database. Custom/self-hosted models need a bare registration first.
    ensure_model_info_registered(cost_cfg)

    prepared = []
    model_roles: dict[str, str] = {}
    for entry, spec in zip(suite_cfg.entries, specs):
        p = prepare(spec, judge=cfg.judges.get(spec.id), epochs=entry.epochs)
        prepared.append((entry, p))
        if p.model_roles:
            model_roles.update(p.model_roles)

    # 스펙 §12: eval_set()은 model 목록 전체에 단일 GenerateConfig만 받는다 —
    # 모델별 max_tokens/temperature/seed가 갈리면 조용히 한쪽 값으로 뭉개질 수
    # 있으므로, 한 run에 섞인 모델의 generate args가 다르면 명시적으로 거부한다.
    gen_args = [generate_args(m) for m in models.values()]
    if gen_args and any(g != gen_args[0] for g in gen_args[1:]):
        raise ValueError(
            "models in one run must share generate args "
            "(max_tokens/temperature/seed); run them separately"
        )
    extra_gen_args = gen_args[0] if gen_args else {}

    success, _logs = eval_set(
        tasks=[p.task for _, p in prepared],
        model=[m.inspect_model for m in models.values()],
        log_dir=str(log_dir),
        token_limit=budget_cfg.token_limit,
        message_limit=budget_cfg.message_limit,
        working_limit=budget_cfg.working_limit_s,
        cost_limit=budget_cfg.cost_limit if cost_cfg else None,
        model_cost_config=cost_cfg or None,
        model_roles=model_roles or None,
        limit=limit,
        fail_on_error=0.05,  # 스펙 §6: 샘플 오류 5% 초과 시 run 실패
        retry_on_error=3,
        **extra_gen_args,
    )

    return RunOutcome(run_dir=run_dir, success=success, log_dir=log_dir)
