from scieval.config import ModelConfig, Pricing
from scieval.providers.registry import (
    ensure_model_info_registered,
    generate_args,
    model_cost_config,
    required_env,
)


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


def test_ensure_model_info_registered_unblocks_set_model_cost():
    """Carry-forward finding from Task 3 review: inspect_ai's set_model_cost()
    raises ValueError for models absent from Inspect's model-info database.
    Custom/self-hosted models (e.g. openai-api/<org>/<model>) need a bare
    ModelInfo registered first so eval_set(model_cost_config=...) -- which
    calls set_model_cost() per entry -- doesn't crash at startup. This is a
    local, offline registration; no network access is involved.
    """
    from inspect_ai.model import ModelCost, get_model_info, set_model_cost

    name = "openai-api/unit-test-registry-fixture/model-9182"
    assert get_model_info(name) is None, "test model must be unknown to Inspect"

    cost = ModelCost(input=0.5, output=1.5, input_cache_write=0.0, input_cache_read=0.0)
    try:
        set_model_cost(name, cost)
    except ValueError as e:
        assert "not found" in str(e)
    else:
        raise AssertionError("expected set_model_cost to fail before registration")

    ensure_model_info_registered({name: cost})
    assert get_model_info(name) is not None

    # this no longer raises now that the model is registered
    set_model_cost(name, cost)
    assert get_model_info(name).cost == cost


def test_ensure_model_info_registered_skips_known_models():
    """A model already in Inspect's database must not be clobbered with a
    bare registration -- only unknown models need the workaround."""
    from inspect_ai.model import ModelCost

    name = "openai/gpt-4o-mini"  # resolvable from Inspect's bundled DB, offline
    cost = ModelCost(input=1.0, output=1.0, input_cache_write=0.0, input_cache_read=0.0)

    ensure_model_info_registered({name: cost})

    from inspect_ai.model import get_model_info

    info = get_model_info(name)
    assert info is not None
    assert info.organization == "OpenAI"  # real DB metadata, not a bare ModelInfo()
