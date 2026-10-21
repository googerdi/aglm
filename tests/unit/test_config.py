from pathlib import Path

import pytest

from aglm.config import (
    AppConfig,
    AttentionConfig,
    ModelConfig,
    TokenizerConfig,
    load_config,
)


def test_default_config_is_valid():
    config = ModelConfig()

    assert config.vocab_size == 4096
    assert config.context_length == 128
    assert config.d_model == 64
    assert config.num_layers == 2
    assert config.num_heads == 4


def test_valid_custom_config():
    config = ModelConfig(
        vocab_size=128,
        context_length=32,
        d_model=48,
        num_layers=3,
        num_heads=4,
    )

    assert config.d_model // config.num_heads == 12


@pytest.mark.parametrize(
    "field,value",
    [
        ("vocab_size", 0),
        ("context_length", 0),
        ("d_model", 0),
        ("num_layers", 0),
        ("num_heads", 0),
    ],
)
def test_non_positive_values_are_rejected(field: str, value: int):
    with pytest.raises(ValueError):
        ModelConfig(**{field: value})


def test_invalid_attention_dimensions_are_rejected():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=63, num_heads=4)


def test_config_is_immutable() -> None:
    config = ModelConfig()

    with pytest.raises(AttributeError):
        config.d_model = 128


@pytest.mark.parametrize(
    (
        "profile",
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "num_kv_heads",
    ),
    [
        ("tiny", 8192, 512, 384, 10, 6, 2),
        ("small", 24576, 2048, 768, 12, 12, 4),
        ("base", 65536, 8192, 1536, 24, 24, 8),
    ],
)
def test_yaml_profiles_load_into_structured_config(
    profile: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
) -> None:
    config = load_config(Path(__file__).parents[2] / "configs" / f"{profile}.yaml")

    assert isinstance(config, AppConfig)
    assert config.profile == profile
    assert config.model == ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        attention=AttentionConfig(num_kv_heads=num_kv_heads),
    )
    assert config.tokenizer.vocab_size == config.model.vocab_size
    assert config.model.attention.pattern == "full_causal"
    assert config.runtime.sdpa_backend == "auto"


def test_load_config_rejects_missing_sections(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("profile: tiny\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing configuration sections"):
        load_config(path)


def test_yaml_declares_vocab_size_only_for_tokenizer() -> None:
    import yaml

    config_path = Path(__file__).parents[2] / "configs" / "tiny.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert "vocab_size" not in raw["model"]
    assert raw["tokenizer"]["vocab_size"] == load_config(config_path).model.vocab_size


def test_tokenizer_identifier_is_an_explicit_registry_key() -> None:
    config = TokenizerConfig(type="byte_level_bpe")

    assert config.type == "byte_level_bpe"


def test_tokenizer_identifier_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="type must not be empty"):
        TokenizerConfig(type="")


@pytest.mark.parametrize(
    "attention_kwargs",
    [
        {"pattern": "sliding_window"},
        {"sliding_window": 0},
        {"layer_overrides": {0: "sliding_window"}},
    ],
)
def test_invalid_attention_window_config_is_rejected(
    attention_kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="sliding_window"):
        AttentionConfig(**attention_kwargs)


def test_gqa_head_counts_must_be_compatible() -> None:
    with pytest.raises(ValueError, match="divisible by num_kv_heads"):
        ModelConfig(
            d_model=96,
            num_heads=6,
            attention=AttentionConfig(num_kv_heads=4),
        )


def test_attention_layer_override_must_be_in_model() -> None:
    with pytest.raises(ValueError, match="less than num_layers"):
        ModelConfig(
            num_layers=2, attention=AttentionConfig(layer_overrides={2: "full_causal"})
        )


def test_sliding_window_hybrid_config_is_valid() -> None:
    config = ModelConfig(
        num_layers=3,
        attention=AttentionConfig(
            pattern="sliding_window",
            sliding_window=256,
            layer_overrides={2: "full_causal"},
        ),
    )

    assert config.attention.layer_overrides[2] == "full_causal"
