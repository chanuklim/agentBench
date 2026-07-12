import math
from dataclasses import dataclass

from scieval.config import WeightsConfig


@dataclass(frozen=True)
class AxisScore:
    score: float
    coverage: float


@dataclass(frozen=True)
class IndexResult:
    value: float
    partial: bool


def axis_scores(normalized: dict[str, float], weights: WeightsConfig) -> dict[str, AxisScore]:
    out: dict[str, AxisScore] = {}
    for axis, aw in weights.axes.items():
        contrib = {b: w for b, w in aw.benchmarks.items() if w > 0 and b in normalized}
        declared = sum(w for w in aw.benchmarks.values() if w > 0)
        if not contrib or declared == 0:
            continue
        total_w = sum(contrib.values())
        score = sum(normalized[b] * w for b, w in contrib.items()) / total_w
        out[axis] = AxisScore(score=score, coverage=total_w / declared)
    return out


def final_index(
    axes: dict[str, AxisScore], weights: WeightsConfig, epsilon: float = 0.01
) -> IndexResult:
    available = {a: s for a, s in axes.items() if a in weights.axes}
    partial = set(available) != set(weights.axes)
    if not available:
        return IndexResult(value=0.0, partial=True)
    total_w = sum(weights.axes[a].weight for a in available)
    log_sum = sum(
        weights.axes[a].weight * math.log(max(s.score, epsilon))
        for a, s in available.items()
    )
    return IndexResult(value=math.exp(log_sum / total_w), partial=partial)
