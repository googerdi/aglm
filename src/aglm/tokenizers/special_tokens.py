from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokenIds:
    eos: int
    bos: int | None = None
    pad: int | None = None
    unk: int | None = None
