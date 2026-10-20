from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096
    context_length: int = 128
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.context_length <= 0:
            raise ValueError("context_length must be positive")

        if self.d_model <= 0:
            raise ValueError("d_model must be positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
