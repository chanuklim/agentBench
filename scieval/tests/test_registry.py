from scieval.config import ModelConfig, Pricing
from scieval.providers.registry import generate_args, model_cost_config, required_env


def _m(**kw):
    return ModelConfig(id="m", inspect_model="openai-api/solar/solar-open-100b", **kw)


def test_required_env(monkeypatch):
    m = _m(base_url_env="SOLAR_BASE_URL", api_key_env="SOLAR_API_KEY")
    monkeypatch.delenv("SOLAR_BASE_URL", raising=False)
    monkeypatch.setenv("SOLAR_API_KEY", "x")
    assert required_env(m) == ["SOLAR_BASE_URL"]


def test_model_cost_config():
    from inspect_ai.model import ModelCost

    m = _m(pricing=Pricing(input_per_mtok=0.5, output_per_mtok=1.5))
    cfg = model_cost_config({"m": m, "nc": ModelConfig(id="nc", inspect_model="mockllm/model")})
    assert set(cfg) == {"openai-api/solar/solar-open-100b"}
    mc = cfg["openai-api/solar/solar-open-100b"]
    assert isinstance(mc, ModelCost)
    assert (mc.input, mc.output, mc.input_cache_write, mc.input_cache_read) == (0.5, 1.5, 0.0, 0.0)


def test_generate_args():
    assert generate_args(_m(max_tokens=64000, seed=7)) == {"max_tokens": 64000, "seed": 7}
    assert generate_args(_m()) == {}
