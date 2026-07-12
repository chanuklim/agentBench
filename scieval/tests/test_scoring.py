import math

import pytest

from scieval.config import Anchor, AxisWeights, WeightsConfig
from scieval.scoring.aggregate import axis_scores, final_index
from scieval.scoring.gates import evaluate
from scieval.scoring.normalize import normalize

W = WeightsConfig(axes={
    "reasoning": AxisWeights(weight=22, benchmarks={"a": 14, "b": 5, "z": 3}),
    "coding": AxisWeights(weight=20, benchmarks={"c": 20}),
})


def test_normalize_expert_anchor():
    a = Anchor(naive=0.25, expert=0.85, quality="sourced")
    assert normalize(0.85, a) == pytest.approx(1.0)
    assert normalize(0.25, a) == 0.0
    assert normalize(0.10, a) == 0.0            # clip 하한
    assert normalize(0.97, a) == pytest.approx(1.2)  # clip 상한


def test_normalize_fallback():
    a = Anchor(naive=0.25, expert=None, quality="fallback")
    assert normalize(0.625, a) == pytest.approx(0.5)


def test_axis_scores_renormalizes_missing():
    # z 결측 → reasoning은 a,b만으로 가중 (14,5)/19
    axes = axis_scores({"a": 1.0, "b": 0.5, "c": 0.8}, W)
    assert axes["reasoning"].score == pytest.approx((14 * 1.0 + 5 * 0.5) / 19)
    assert axes["reasoning"].coverage == pytest.approx(19 / 22)
    assert axes["coding"].coverage == 1.0


def test_final_index_geometric():
    axes = axis_scores({"a": 0.9, "b": 0.9, "z": 0.9, "c": 0.4}, W)
    r = final_index(axes, W)
    expected = math.exp((22 * math.log(0.9) + 20 * math.log(0.4)) / 42)
    assert r.value == pytest.approx(expected)
    assert r.partial is False


def test_final_index_epsilon_floor_and_partial():
    axes = axis_scores({"a": 0.0, "b": 0.0, "z": 0.0}, W)  # coding 축 결측
    r = final_index(axes, W)
    assert r.partial is True
    assert r.value == pytest.approx(0.01)  # ε 하한


def test_gates():
    axes = axis_scores({"a": 0.9, "b": 0.9, "z": 0.9, "c": 0.2}, W)
    out = evaluate(axes, {"reasoning": 0.5, "coding": 0.5, "literature": 0.3, "tooluse": None})
    assert out == {"reasoning": "pass", "coding": "fail",
                   "literature": "missing", "tooluse": "disabled"}
