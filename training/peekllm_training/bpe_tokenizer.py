from dataclasses import dataclass, field

import tiktoken

END_OF_TEXT_TOKEN = "<|endoftext|>"


@dataclass
class BPETokenizer:
    """Encodes/decodes text using GPT-2's byte pair encoding vocabulary."""

    encoding_name: str = "gpt2"
    _encoding: tiktoken.Encoding = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._encoding = tiktoken.get_encoding(self.encoding_name)

    @property
    def vocab_size(self) -> int:
        return self._encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, allowed_special={END_OF_TEXT_TOKEN})

    def decode(self, ids: list[int]) -> str:
        return self._encoding.decode(ids)
