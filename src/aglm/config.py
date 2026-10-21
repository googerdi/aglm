from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AttentionConfig:
    implementation: str = "gqa"
    pattern: str = "full_causal"
    num_kv_heads: int = 2
    sliding_window: int | None = None
    layer_overrides: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.implementation:
            raise ValueError("attention implementation must not be empty")
        if self.pattern not in {"full_causal", "sliding_window"}:
            raise ValueError("attention pattern must be full_causal or sliding_window")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive when set")
        if self.pattern == "sliding_window" and self.sliding_window is None:
            raise ValueError("sliding_window is required for sliding_window pattern")
        for layer_index, pattern in self.layer_overrides.items():
            if not isinstance(layer_index, int) or layer_index < 0:
                raise ValueError(
                    "attention layer override indices must be non-negative integers"
                )
            if pattern not in {"full_causal", "sliding_window"}:
                raise ValueError(
                    "attention layer override pattern must be full_causal or sliding_window"
                )
            if pattern == "sliding_window" and self.sliding_window is None:
                raise ValueError("sliding_window is required by a layer override")


@dataclass(frozen=True)
class RuntimeConfig:
    sdpa_backend: str = "auto"

    def __post_init__(self) -> None:
        if self.sdpa_backend not in {"auto", "flash", "efficient", "math", "cudnn"}:
            raise ValueError(
                "sdpa_backend must be auto, flash, efficient, math, or cudnn"
            )


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096
    context_length: int = 128
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    attention: AttentionConfig = field(default_factory=AttentionConfig)

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
        if self.num_heads % self.attention.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        invalid_layers = set(self.attention.layer_overrides) - set(
            range(self.num_layers)
        )
        if invalid_layers:
            raise ValueError(
                "attention layer override index must be less than num_layers"
            )


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
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

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
        model_raw = dict(raw["model"])
        attention = AttentionConfig(**model_raw.pop("attention", {}))
        model = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            attention=attention,
            **model_raw,
        )
        return AppConfig(
            profile=raw["profile"],
            model=model,
            tokenizer=tokenizer,
            training=TrainingConfig(**raw["training"]),
            hardware=HardwareConfig(**raw["hardware"]),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
        )
    except TypeError as exc:
        raise ValueError(f"invalid configuration fields: {exc}") from exc
