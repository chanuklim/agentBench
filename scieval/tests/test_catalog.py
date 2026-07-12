import pytest

from scieval.tasks.catalog import CATALOG
from scieval.tasks.spec import BenchmarkSpec, prepare


def test_catalog_entries():
    assert set(CATALOG) == {"gpqa_diamond", "hle_stem", "frontierscience", "scicode"}
    assert CATALOG["scicode"].requires == {"docker"}
    assert CATALOG["hle_stem"].gated is True
    assert CATALOG["gpqa_diamond"].axis == "reasoning"
    assert CATALOG["scicode"].axis == "coding"


def test_judge_required():
    calls: list[dict] = []
    s = BenchmarkSpec(
        id="x", axis="reasoning", task_name="x", agentic=False, gated=False,
        requires=set(), default_epochs=1, judge_mode="task_param_grader_model",
        dataset_revision=None, loader=lambda **kw: calls.append(kw) or "TASK",
    )
    with pytest.raises(ValueError, match="judge"):
        prepare(s, judge=None, epochs=None)
    p = prepare(s, judge="openai/gpt-4o-mini", epochs=None)
    assert p.task == "TASK"
    assert calls[0]["grader_model"] == "openai/gpt-4o-mini"
    assert p.model_roles is None


def test_judge_model_role():
    s = BenchmarkSpec(
        id="x", axis="reasoning", task_name="x", agentic=False, gated=False,
        requires=set(), default_epochs=1, judge_mode="model_role_grader",
        dataset_revision=None, loader=lambda **kw: "TASK",
    )
    p = prepare(s, judge="openai/gpt-4o-mini", epochs=None)
    assert p.model_roles == {"grader": "openai/gpt-4o-mini"}


def test_no_judge_benchmark():
    s = CATALOG["gpqa_diamond"]
    assert s.judge_mode == "none"
