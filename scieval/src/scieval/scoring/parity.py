import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ParityVerdict:
    ok: bool
    diff: float
    tolerance: float


def parity_check(observed: float, n: int, published: float, tol_pp: float = 0.03) -> ParityVerdict:
    se = math.sqrt(max(observed * (1.0 - observed), 1e-12) / n)
    tolerance = max(tol_pp, 1.96 * se)
    diff = abs(observed - published)
    return ParityVerdict(ok=diff <= tolerance, diff=diff, tolerance=tolerance)
