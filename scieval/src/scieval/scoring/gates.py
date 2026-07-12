from scieval.scoring.aggregate import AxisScore


def evaluate(axes: dict[str, AxisScore], thresholds: dict[str, float | None]) -> dict[str, str]:
    out: dict[str, str] = {}
    for axis, threshold in thresholds.items():
        if threshold is None:
            out[axis] = "disabled"
        elif axis not in axes:
            out[axis] = "missing"
        else:
            out[axis] = "pass" if axes[axis].score >= threshold else "fail"
    return out
