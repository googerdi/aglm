import pytest

from aglm import registry
from aglm.config import TokenizerConfig
from aglm.registry import create_tokenizer


def test_create_tokenizer_rejects_unregistered_identifier() -> None:
    with pytest.raises(KeyError, match="unknown tokenizer identifier"):
        create_tokenizer(TokenizerConfig(type="not_registered"))


def test_builtin_tokenizer_identifier_resolves_to_bpe_module() -> None:
    assert registry._TOKENIZER_MODULES["byte_level_bpe"] == "aglm.tokenizers.bpe"


def test_create_tokenizer_imports_and_instantiates_registered_class(
    monkeypatch,
) -> None:
    class StubTokenizer:
        def __init__(self, *, config: TokenizerConfig, marker: str) -> None:
            self.config = config
            self.marker = marker

    monkeypatch.setitem(registry._TOKENIZER_MODULES, "stub", "tests.stub_tokenizer")
    monkeypatch.setitem(registry._TOKENIZERS, "stub", StubTokenizer)
    imported_modules: list[str] = []
    monkeypatch.setattr(
        registry,
        "import_module",
        lambda module: imported_modules.append(module),
    )
    config = TokenizerConfig(type="stub", vocab_size=128)

    tokenizer = create_tokenizer(config, marker="loaded")

    assert imported_modules == ["tests.stub_tokenizer"]
    assert tokenizer.config is config
    assert tokenizer.marker == "loaded"


def test_register_tokenizer_rejects_duplicate_registration(monkeypatch) -> None:
    monkeypatch.setitem(registry._TOKENIZER_MODULES, "duplicate", "tests.duplicate")

    @registry.register_tokenizer("duplicate")
    class FirstTokenizer:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @registry.register_tokenizer("duplicate")
        class SecondTokenizer:
            pass

    assert registry._TOKENIZERS["duplicate"] is FirstTokenizer
