from pathlib import Path

from aglm.tokenizers.base import Tokenizer
from aglm.tokenizers.special_tokens import SpecialTokenIds


class DummyTokenizer:
    """A minimal tokenizer used to test the public contract."""

    def __init__(self) -> None:
        self._special_tokens = SpecialTokenIds(eos=9)

    @property
    def vocab_size(self) -> int:
        return 10

    @property
    def special_tokens(self) -> SpecialTokenIds:
        return self._special_tokens

    def encode(
        self,
        text: str,
        *,
        add_eos: bool = False,
    ) -> list[int]:
        token_ids = [1, 2] if text else []

        if add_eos:
            token_ids.append(self.special_tokens.eos)

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        if skip_special_tokens:
            token_ids = [
                token_id
                for token_id in token_ids
                if token_id != self.special_tokens.eos
            ]

        return "hello" if token_ids == [1, 2] else ""

    def save(self, path: str | Path) -> None:
        pass


def test_tokenizer_exposes_vocabulary_size():
    tokenizer: Tokenizer = DummyTokenizer()

    assert tokenizer.vocab_size == 10


def test_tokenizer_exposes_special_token_ids():
    tokenizer: Tokenizer = DummyTokenizer()

    assert tokenizer.special_tokens.eos == 9


def test_encode_can_append_eos():
    tokenizer: Tokenizer = DummyTokenizer()

    assert tokenizer.encode("hello") == [1, 2]
    assert tokenizer.encode("hello", add_eos=True) == [1, 2, 9]


def test_decode_can_skip_special_tokens():
    tokenizer: Tokenizer = DummyTokenizer()

    assert tokenizer.decode([1, 2, 9]) == "hello"
    assert (
        tokenizer.decode(
            [1, 2, 9],
            skip_special_tokens=False,
        )
        == ""
    )


def test_empty_text_produces_no_regular_tokens():
    tokenizer: Tokenizer = DummyTokenizer()

    assert tokenizer.encode("") == []
    assert tokenizer.encode("", add_eos=True) == [9]
