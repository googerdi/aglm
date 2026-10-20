from pathlib import Path
from typing import Protocol

from aglm.tokenizers.special_tokens import SpecialTokenIds


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int:
        """Number of all valid token IDs, including special tokens."""
        ...

    @property
    def special_tokens(self) -> SpecialTokenIds:
        """IDs assigned to special tokens."""
        ...

    def encode(
        self,
        text: str,
        *,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text and optionally append the EOS token."""
        ...

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs into text."""
        ...

    def save(self, path: str | Path) -> None:
        """Save tokenizer vocabulary and configuration."""
        ...
