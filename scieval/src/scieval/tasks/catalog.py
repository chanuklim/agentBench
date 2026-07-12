"""M1 benchmark catalog.

inspect_evals 임포트는 loader 안에서만 한다(모듈 import가 데이터셋
다운로드를 트리거하지 않도록). dataset_revision은 inspect_evals 0.14.3이
핀한 값을 기록용으로 복사한 것이다.
"""
from scieval.tasks.spec import BenchmarkSpec

# HLE metadata "category" 필드 기준 STEM 필터. 구현 시 첫 fetch 후
# 실제 category 고유값과 대조해 검증할 것 (runbook Task 12 참조).
HLE_STEM_CATEGORIES = [
    "Math",
    "Physics",
    "Chemistry",
    "Biology/Medicine",
    "Engineering",
    "Computer Science/AI",
]


def _load_gpqa(epochs: int | None = None, **kw):
    from inspect_evals.gpqa import gpqa_diamond

    return gpqa_diamond(epochs=epochs) if epochs is not None else gpqa_diamond()


def _load_hle(**kw):
    from inspect_evals.hle import hle

    return hle(include_multi_modal=False, category=HLE_STEM_CATEGORIES)


def _load_frontierscience(grader_model: str | None = None, **kw):
    from inspect_evals.frontierscience import frontierscience

    return frontierscience(grader_model=grader_model)


def _load_scicode(**kw):
    from inspect_evals.scicode import scicode

    return scicode(timeout=300)


CATALOG: dict[str, BenchmarkSpec] = {
    "gpqa_diamond": BenchmarkSpec(
        id="gpqa_diamond", axis="reasoning", task_name="gpqa_diamond",
        agentic=False, gated=False, requires=set(), default_epochs=4,
        judge_mode="none", dataset_revision=None, loader=_load_gpqa,
        supports_epochs=True,
    ),
    "hle_stem": BenchmarkSpec(
        id="hle_stem", axis="reasoning", task_name="hle",
        agentic=False, gated=True, requires=set(), default_epochs=1,
        judge_mode="model_role_grader",
        dataset_revision="5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
        loader=_load_hle,
    ),
    "frontierscience": BenchmarkSpec(
        id="frontierscience", axis="reasoning", task_name="frontierscience",
        agentic=False, gated=False, requires=set(), default_epochs=1,
        judge_mode="task_param_grader_model",
        dataset_revision="25ed67db7da8f4591484e764008ff585544f5a30",
        loader=_load_frontierscience,
    ),
    "scicode": BenchmarkSpec(
        id="scicode", axis="coding", task_name="scicode",
        agentic=False, gated=False, requires={"docker"}, default_epochs=1,
        judge_mode="none", dataset_revision=None, loader=_load_scicode,
    ),
}
