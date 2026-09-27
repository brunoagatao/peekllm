import re
from dataclasses import dataclass, field
from typing import ClassVar

# Regex used to split raw text into tokens. Captures common punctuation and
# the em dash ("--") as their own tokens, and splits on any whitespace run.
_SPLIT_PATTERN = re.compile(r'([,.:;?_!"()\']|--|\s)')

# Regex used during decode() to remove the space that build_vocab/encode()
# inserts before punctuation, so "hello , world !" becomes "hello, world!".
_PUNCTUATION_SPACING_PATTERN = re.compile(r'\s+([,.?!"()\'])')


def tokenize(text: str) -> list[str]:
    """Split raw text into a list of word/punctuation tokens.

    Splits on whitespace and a fixed set of punctuation characters, keeping
    the punctuation as its own token, and drops any empty or whitespace-only
    fragments produced by the split.
    """
    fragments = _SPLIT_PATTERN.split(text)
    return [fragment.strip() for fragment in fragments if fragment.strip()]


def build_vocab(text: str) -> dict[str, int]:
    """Build a token -> id vocabulary from raw text.

    IDs are assigned in sorted order of the unique tokens, so the mapping is
    deterministic for a given input text.
    """
    tokens = tokenize(text)
    unique_tokens = sorted(set(tokens))
    return {token: idx for (idx, token) in enumerate(unique_tokens)}


@dataclass
class SimpleTokenizerV1:
    """Encodes/decodes text using a fixed vocabulary.

    Raises a `KeyError` on `encode()` if the text contains a token that is not
    in the vocabulary. See `SimpleTokenizerV2` for a version that instead maps
    unknown tokens to an "<|unk|>" placeholder.
    """

    str_to_int: dict[str, int]
    int_to_str: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.int_to_str = {idx: token for (token, idx) in self.str_to_int.items()}

    def encode(self, text: str) -> list[int]:
        tokens = tokenize(text)
        return [self.str_to_int[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[idx] for idx in ids)
        return _PUNCTUATION_SPACING_PATTERN.sub(r"\1", text)


@dataclass
class SimpleTokenizerV2:
    """Like SimpleTokenizerV1, but tolerates tokens outside the vocabulary.

    Adds two special tokens to whatever vocabulary is passed in:
    -  "<|unk|>": substituted for any token not seen during vocabulary
    construction, so `encode()` never raises a `KeyError`.
    -  "<|endoftext|>": a separator that can be used to join independent
    documents/chunks of text before feeding them to a model.
    """

    str_to_int: dict[str, int]
    int_to_str: dict[int, str] = field(init=False)

    UNKNOWN_TOKEN: ClassVar[str] = "<|unk|>"
    END_OF_TEXT_TOKEN: ClassVar[str] = "<|endoftext|>"

    def __post_init__(self) -> None:
        # Copy so we don't mutate a vocab dict the caller might still hold a
        # reference to (SimpleTokenizerV1 aliases it directly; this class
        # adds entries, so it needs its own copy).
        extended_vocab = dict(self.str_to_int)
        for special_token in (self.UNKNOWN_TOKEN, self.END_OF_TEXT_TOKEN):
            if special_token not in extended_vocab:
                extended_vocab[special_token] = len(extended_vocab)

        self.str_to_int = extended_vocab
        self.int_to_str = {idx: token for (token, idx) in extended_vocab.items()}

    def encode(self, text: str) -> list[int]:
        tokens = tokenize(text)
        tokens = [token if token in self.str_to_int else self.UNKNOWN_TOKEN for token in tokens]
        return [self.str_to_int[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[idx] for idx in ids)
        return _PUNCTUATION_SPACING_PATTERN.sub(r"\1", text)
