#!/usr/bin/env python
"""Smoke checks for the config-gated residual Local Detail Adapter (LDA).

Covers both MoonViT modeling copies (train import path and utils/worker import
path) plus all four MoonViTConfig classes. Model builds use a reduced encoder
(num_hidden_layers / intermediate_size / init_pos_emb_*) because LDA presence
and its parameter count depend only on hidden_size and the LDA config fields.
Run from Embodied/: python scripts/smoke_lda.py
"""
import torch

from eaglevl.model.locany.configuration_locateanything import (
    MoonViTConfig as TrainLocanyViTConfig,
)
from eaglevl.model.moon_vit.modeling_vit import (
    LocalDetailAdapter as TrainLocalDetailAdapter,
    MoonViTConfig as TrainViTConfig,
    MoonVitPretrainedModel as TrainMoonViT,
)
from eaglevl.utils.locany.configuration_locateanything import (
    MoonViTConfig as UtilsLocanyViTConfig,
)
from eaglevl.utils.locany.modeling_vit import (
    LocalDetailAdapter as UtilsLocalDetailAdapter,
    MoonViTConfig as UtilsViTConfig,
    MoonVitPretrainedModel as UtilsMoonViT,
)

LDA_VALUE_ERROR = "lda_bottleneck_dim must be a positive integer"
LDA_PARAM_COUNT = 1152 * 128 + 128 + 128 * 9 + 128 + 128 * 1152 + 1152 + 1  # 297473


def small_config(config_cls, **overrides):
    kwargs = dict(
        num_hidden_layers=2,
        intermediate_size=64,
        init_pos_emb_height=8,
        init_pos_emb_width=8,
    )
    kwargs.update(overrides)
    cfg = config_cls(**kwargs)
    cfg._attn_implementation = "sdpa"
    return cfg


def assert_module_tree(label, model_cls, config_cls):
    default_cfg = config_cls()
    assert default_cfg.use_lda is False, (label, "default use_lda")
    assert default_cfg.lda_bottleneck_dim == 128, (label, "default lda_bottleneck_dim")
    for cfg in (small_config(config_cls), small_config(config_cls, use_lda=False)):
        model = model_cls(cfg)
        assert getattr(model, "lda", None) is None, (label, "lda attribute present while off")
        names = [name for name, _ in model.named_parameters()]
        assert not any(".lda." in name for name in names), (label, names)
        del model


def assert_module_on(label, model_cls, config_cls):
    cfg = small_config(config_cls, use_lda=True, lda_bottleneck_dim=128)
    model = model_cls(cfg)
    lda = getattr(model, "lda", None)
    assert lda is not None, (label, "lda missing while on")
    assert float(lda.gamma.detach()) == 0.0, (label, "gamma not zero")
    n = sum(p.numel() for p in lda.parameters())
    assert n == LDA_PARAM_COUNT, (label, n)
    del model


def assert_adapter(label, adapter_cls):
    torch.manual_seed(0)
    hidden = torch.randn(1024, 1152)
    grid = torch.tensor([[32, 32]])
    adapter = adapter_cls(1152, 128).eval()

    with torch.no_grad():
        out = adapter(hidden, grid)
    assert torch.allclose(out, hidden, atol=0, rtol=0), (label, "gamma=0 not identity")

    packed = torch.randn(768, 1152)
    with torch.no_grad():
        out_packed = adapter(packed, torch.tensor([[16, 16], [32, 16]]))
    assert torch.allclose(out_packed, packed, atol=0, rtol=0), (label, "packed gamma=0 not identity")

    adapter.gamma.data.fill_(1.0)
    with torch.no_grad():
        out1 = adapter(hidden, grid)
    assert not torch.allclose(out1, hidden, atol=1e-5), (label, "gamma=1 output equals input")
    # Sliced per-image conv must be independent of packing context.
    with torch.no_grad():
        out_packed1 = adapter(packed, torch.tensor([[16, 16], [32, 16]]))
    solo = packed[:256].clone()
    with torch.no_grad():
        out_solo = adapter(solo, torch.tensor([[16, 16]]))
    assert torch.allclose(out_packed1[:256], out_solo, atol=1e-5), (label, "packed/solo mismatch")

    with torch.no_grad():
        try:
            adapter(torch.randn(100, 1152), torch.tensor([[20, 20]]))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{label}: oversubscribed grid did not raise")
        try:
            adapter(torch.randn(100, 1152), torch.tensor([[5, 5]]))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{label}: leftover tokens did not raise")

    try:
        adapter_cls(1152, 0)
    except ValueError as exc:
        assert str(exc) == LDA_VALUE_ERROR, (label, str(exc))
    else:
        raise AssertionError(f"{label}: LocalDetailAdapter(1152, 0) did not raise")


def assert_config_roundtrip(label, config_cls):
    d = config_cls(use_lda=True, lda_bottleneck_dim=64).to_dict()
    assert d["use_lda"] is True, (label, d.get("use_lda"))
    assert d["lda_bottleneck_dim"] == 64, (label, d.get("lda_bottleneck_dim"))
    rebuilt = config_cls(**d)
    assert rebuilt.use_lda is True, (label, "rebuilt use_lda")
    assert rebuilt.lda_bottleneck_dim == 64, (label, "rebuilt lda_bottleneck_dim")


def parse_optional_bool(value, name):
    # Mirrors the main()-local _parse_optional_bool in locany_finetune_magi_stream.py.
    if value is None or str(value).strip() == "":
        return None
    v = str(value).strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ValueError(f"{name} must be true or false")


def assert_parse_optional_bool():
    assert parse_optional_bool(None, "use_lda") is None
    assert parse_optional_bool("", "use_lda") is None
    for truthy in ("true", "1", "yes", "True", "YES"):
        assert parse_optional_bool(truthy, "use_lda") is True, truthy
    for falsy in ("false", "0", "no", "False", "NO"):
        assert parse_optional_bool(falsy, "use_lda") is False, falsy
    try:
        parse_optional_bool("maybe", "use_lda")
    except ValueError as exc:
        assert "use_lda must be true or false" in str(exc)
    else:
        raise AssertionError("parse_optional_bool('maybe') did not raise")


def main():
    for label, model_cls, config_cls, adapter_cls in [
        ("train", TrainMoonViT, TrainViTConfig, TrainLocalDetailAdapter),
        ("utils", UtilsMoonViT, UtilsViTConfig, UtilsLocalDetailAdapter),
    ]:
        assert_module_tree(label, model_cls, config_cls)
        assert_module_on(label, model_cls, config_cls)
        assert_adapter(label, adapter_cls)

    for label, config_cls in [
        ("train-vit", TrainViTConfig),
        ("utils-vit", UtilsViTConfig),
        ("train-locany", TrainLocanyViTConfig),
        ("utils-locany", UtilsLocanyViTConfig),
    ]:
        assert_config_roundtrip(label, config_cls)

    assert_parse_optional_bool()
    print("lda smoke passed")


if __name__ == "__main__":
    main()
