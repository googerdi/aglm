import pytest

from aglm.config import ModelConfig


def test_default_config_is_valid():
    config = ModelConfig()

    assert config.vocab_size == 1024
    assert config.context_length == 64
    assert config.d_model == 64
    assert config.num_layers == 2
    assert config.num_heads == 4
    assert config.d_model % config.num_heads == 0


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
