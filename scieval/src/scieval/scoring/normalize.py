from scieval.config import Anchor


def normalize(raw: float, anchor: Anchor) -> float:
    hi = anchor.expert if anchor.expert is not None else 1.0
    span = hi - anchor.naive
    if span <= 0:
        raise ValueError("anchor expert/upper bound must exceed naive baseline")
    return min(max((raw - anchor.naive) / span, 0.0), 1.2)
