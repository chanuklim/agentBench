from dataclasses import dataclass
from typing import Any, Callable, Literal

JudgeMode = Literal["none", "model_role_grader", "task_param_grader_model"]


@dataclass(frozen=True)
class BenchmarkSpec:
    id: str
    axis: str
    task_name: str          # Inspect 로그의 task 이름 (결과 매칭용)
    agentic: bool
    gated: bool             # HF gated 데이터셋 여부 (HF_TOKEN 필요)
    requires: set[str]      # 프로파일 provides와 대조 (예: {"docker"})
    default_epochs: int
    judge_mode: JudgeMode
    dataset_revision: str | None
    loader: Callable[..., Any]   # kwargs를 받아 inspect Task 반환


@dataclass(frozen=True)
class PreparedBenchmark:
    spec: BenchmarkSpec
    task: Any
    model_roles: dict[str, str] | None


def prepare(s: BenchmarkSpec, judge: str | None, epochs: int | None) -> PreparedBenchmark:
    if s.judge_mode != "none" and judge is None:
        raise ValueError(f"benchmark {s.id} requires a pinned judge model (no self-judging)")
    kwargs: dict[str, Any] = {}
    if epochs is not None:
        kwargs["epochs"] = epochs
    model_roles: dict[str, str] | None = None
    if s.judge_mode == "task_param_grader_model":
        kwargs["grader_model"] = judge
    elif s.judge_mode == "model_role_grader":
        model_roles = {"grader": judge}  # type: ignore[dict-item]
    return PreparedBenchmark(spec=s, task=s.loader(**kwargs), model_roles=model_roles)
