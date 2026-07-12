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


def test_gpqa_supports_epochs():
    assert CATALOG["gpqa_diamond"].supports_epochs is True


def test_only_gpqa_supports_epochs():
    non_epochs_ids = {"hle_stem", "frontierscience", "scicode"}
    for bid in non_epochs_ids:
        assert CATALOG[bid].supports_epochs is False


def test_epochs_override_rejected_when_unsupported():
    s = BenchmarkSpec(
        id="x", axis="reasoning", task_name="x", agentic=False, gated=False,
        requires=set(), default_epochs=1, judge_mode="none",
        dataset_revision=None, loader=lambda **kw: "TASK",
    )
    with pytest.raises(ValueError, match="does not support an epochs override"):
        prepare(s, judge=None, epochs=3)


def test_epochs_override_allowed_when_supported():
    calls: list[dict] = []
    s = BenchmarkSpec(
        id="x", axis="reasoning", task_name="x", agentic=False, gated=False,
        requires=set(), default_epochs=1, judge_mode="none",
        dataset_revision=None, loader=lambda **kw: calls.append(kw) or "TASK",
        supports_epochs=True,
    )
    p = prepare(s, judge=None, epochs=3)
    assert p.task == "TASK"
    assert calls[0]["epochs"] == 3


def test_epochs_none_never_rejected():
    """epochs=None must never trip the strictness check, even on a spec
    that doesn't support epochs overrides."""
    s = BenchmarkSpec(
        id="x", axis="reasoning", task_name="x", agentic=False, gated=False,
        requires=set(), default_epochs=1, judge_mode="none",
        dataset_revision=None, loader=lambda **kw: "TASK",
    )
    p = prepare(s, judge=None, epochs=None)
    assert p.task == "TASK"
