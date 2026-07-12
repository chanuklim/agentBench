import os

from inspect_ai.model import ModelCost

from scieval.config import ModelConfig


def required_env(m: ModelConfig) -> list[str]:
    declared = [e for e in (m.base_url_env, m.api_key_env) if e]
    return [e for e in declared if not os.environ.get(e)]


def model_cost_config(models: dict[str, ModelConfig]) -> dict[str, ModelCost]:
    out: dict[str, ModelCost] = {}
    for m in models.values():
        if m.pricing is not None:
            out[m.inspect_model] = ModelCost(
                input=m.pricing.input_per_mtok,
                output=m.pricing.output_per_mtok,
                input_cache_write=m.pricing.cache_write_per_mtok,
                input_cache_read=m.pricing.cache_read_per_mtok,
            )
    return out


def generate_args(m: ModelConfig) -> dict:
    args = {"max_tokens": m.max_tokens, "temperature": m.temperature, "seed": m.seed}
    return {k: v for k, v in args.items() if v is not None}
