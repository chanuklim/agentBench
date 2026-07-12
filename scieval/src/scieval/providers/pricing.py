from scieval.config import Pricing


def gpu_price_per_mtok(gpu_hour_usd: float, num_gpus: int, throughput_tok_per_s: float) -> float:
    tokens_per_hour = throughput_tok_per_s * 3600.0
    return gpu_hour_usd * num_gpus / tokens_per_hour * 1_000_000.0


def usage_cost(
    p: Pricing, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0
) -> float:
    return (
        input_tokens * p.input_per_mtok
        + output_tokens * p.output_per_mtok
        + cache_read * p.cache_read_per_mtok
        + cache_write * p.cache_write_per_mtok
    ) / 1_000_000.0
