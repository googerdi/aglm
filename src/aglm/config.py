from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096
    context_length: int = 128
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "context_length",
            "d_model",
            "num_layers",
            "num_heads",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")


@dataclass(frozen=True)
class TokenizerConfig:
    type: str = "byte_level_bpe"
    vocab_size: int = 4096

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("tokenizer type must not be empty")
        if self.vocab_size <= 0:
            raise ValueError("tokenizer vocab_size must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    precision: str = "fp32"
    gradient_accumulation_steps: int = 1
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of: fp32, fp16, bf16")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")


@dataclass(frozen=True)
class HardwareConfig:
    target: str = "cpu_or_laptop"

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("hardware target must not be empty")


@dataclass(frozen=True)
class AppConfig:
    profile: str
    model: ModelConfig
    tokenizer: TokenizerConfig
    training: TrainingConfig
    hardware: HardwareConfig

    def __post_init__(self) -> None:
        if not self.profile:
            raise ValueError("profile must not be empty")


def load_config(path: str | Path) -> AppConfig:
    """Load a complete AGLM configuration from a YAML file."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise TypeError("configuration root must be a YAML mapping")

    required = {"profile", "model", "tokenizer", "training", "hardware"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(
            f"missing configuration sections: {', '.join(sorted(missing))}"
        )

    try:
        tokenizer = TokenizerConfig(**raw["tokenizer"])
        model = ModelConfig(vocab_size=tokenizer.vocab_size, **raw["model"])
        return AppConfig(
            profile=raw["profile"],
            model=model,
            tokenizer=tokenizer,
            training=TrainingConfig(**raw["training"]),
            hardware=HardwareConfig(**raw["hardware"]),
        )
    except TypeError as exc:
        raise ValueError(f"invalid configuration fields: {exc}") from exc
