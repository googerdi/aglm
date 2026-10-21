from collections.abc import Callable
from importlib import import_module
from typing import Any, TypeVar

from aglm.config import TokenizerConfig

T = TypeVar("T")

_TOKENIZER_MODULES = {"byte_level_bpe": "aglm.tokenizers.bpe"}
_TOKENIZERS: dict[str, type[Any]] = {}


def register_tokenizer(name: str) -> Callable[[type[T]], type[T]]:
    """Register a tokenizer implementation under its stable YAML identifier."""
    if name not in _TOKENIZER_MODULES:
        raise KeyError(f"unknown tokenizer identifier: {name}")

    def decorator(tokenizer_class: type[T]) -> type[T]:
        if name in _TOKENIZERS:
            raise ValueError(f"tokenizer is already registered: {name}")
        _TOKENIZERS[name] = tokenizer_class
        return tokenizer_class

    return decorator


def create_tokenizer(config: TokenizerConfig, **kwargs: Any) -> Any:
    """Load and instantiate the tokenizer selected by its YAML identifier."""
    name = config.type
    if name not in _TOKENIZER_MODULES:
        raise KeyError(f"unknown tokenizer identifier: {name}")

    import_module(_TOKENIZER_MODULES[name])
    try:
        tokenizer_class = _TOKENIZERS[name]
    except KeyError as exc:
        raise RuntimeError(
            f"tokenizer module for {name!r} did not register an implementation"
        ) from exc
    return tokenizer_class(config=config, **kwargs)
