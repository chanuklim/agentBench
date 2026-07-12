import pytest

from scieval.config import Pricing
from scieval.providers.pricing import gpu_price_per_mtok, usage_cost


def test_gpu_price_per_mtok():
    # $4/GPU-h × 4 GPU, 2000 tok/s → $16/h ÷ 7.2M tok/h × 1M ≈ $2.2222/Mtok
    assert gpu_price_per_mtok(4.0, 4, 2000.0) == pytest.approx(2.2222, abs=1e-3)


def test_usage_cost():
    p = Pricing(input_per_mtok=2.0, output_per_mtok=10.0, cache_read_per_mtok=1.0)
    # 500k input + 100k output + 200k cache_read = 1.0 + 1.0 + 0.2
    assert usage_cost(p, 500_000, 100_000, cache_read=200_000) == pytest.approx(2.2)
