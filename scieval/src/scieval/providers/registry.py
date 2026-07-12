import os

from inspect_ai.model import ModelCost, ModelInfo, get_model_info, set_model_info

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


def ensure_model_info_registered(cost_cfg: dict[str, ModelCost]) -> None:
    """Register a bare `ModelInfo` for any priced model Inspect doesn't know.

    `eval_set(model_cost_config=cost_cfg)` internally calls
    `inspect_ai.model.set_model_cost(name, cost)` for every entry, which
    raises `ValueError("Model '<name>' not found.")` for models absent from
    Inspect's built-in model-info database (e.g. custom self-hosted models
    like `openai-api/solar/solar-open-100b`). Registering a blank
    `ModelInfo()` up front makes the model resolvable so `set_model_cost`
    can attach cost data to it instead of crashing. This is a local,
    in-process registration — no network access involved.
    """
    for name in cost_cfg:
        if get_model_info(name) is None:
            set_model_info(name, ModelInfo())


def generate_args(m: ModelConfig) -> dict:
    args = {"max_tokens": m.max_tokens, "temperature": m.temperature, "seed": m.seed}
    return {k: v for k, v in args.items() if v is not None}
