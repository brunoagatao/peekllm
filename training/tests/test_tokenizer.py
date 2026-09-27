from pathlib import Path

import pytest

from peekllm_training.tokenizer import (
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocab,
    tokenize,
)

VERDICT_PATH = Path(__file__).parent.parent / "peekllm_training" / "data" / "the-verdict.txt"


@pytest.fixture(scope="module")
def verdict_text() -> str:
    return VERDICT_PATH.read_text()


@pytest.fixture(scope="module")
def verdict_vocab(verdict_text: str) -> dict[str, int]:
    return build_vocab(verdict_text)


class TestTokenize:
    def test_splits_words_on_whitespace(self) -> None:
        assert tokenize("Hello world") == ["Hello", "world"]

    def test_splits_punctuation_into_own_tokens(self) -> None:
        assert tokenize("Hello, world!") == ["Hello", ",", "world", "!"]

    def test_splits_em_dash_into_its_own_token(self) -> None:
        assert tokenize("genius--though") == ["genius", "--", "though"]

    def test_handles_multiple_consecutive_whitespace(self) -> None:
        assert tokenize("Hello   world") == ["Hello", "world"]

    def test_drops_no_real_content_on_empty_string(self) -> None:
        assert tokenize("") == []

    def test_drops_no_real_content_on_whitespace_only_string(self) -> None:
        assert tokenize("   \n\t  ") == []

    def test_handles_punctuation_only_input(self) -> None:
        assert tokenize("...!?") == [".", ".", ".", "!", "?"]

    def test_never_produces_empty_or_whitespace_tokens(self) -> None:
        tokens = tokenize("Hello,  world.  Foo--bar!  ")
        assert all(token and not token.isspace() for token in tokens)

    def test_on_verdict_text_matches_expected_scale(self, verdict_text: str) -> None:
        # Sanity check against the book's reported token count for this
        # exact text (allowing a little slack for punctuation edge cases).
        tokens = tokenize(verdict_text)
        assert 4500 <= len(tokens) <= 4800


class TestBuildVocab:
    def test_assigns_unique_sequential_ids(self) -> None:
        vocab = build_vocab("a b c")
        assert sorted(vocab.values()) == list(range(len(vocab)))

    def test_ids_follow_sorted_token_order(self) -> None:
        vocab = build_vocab("banana apple cherry")
        assert vocab["apple"] < vocab["banana"] < vocab["cherry"]

    def test_deterministic_across_calls(self) -> None:
        text = "the quick brown fox jumps over the lazy dog"
        assert build_vocab(text) == build_vocab(text)

    def test_deduplicates_repeated_tokens(self) -> None:
        vocab = build_vocab("the the the cat cat sat")
        assert len(vocab) == len(set(["the", "cat", "sat"]))

    def test_on_verdict_text_matches_expected_scale(self, verdict_vocab: dict[str, int]) -> None:
        assert 1000 <= len(verdict_vocab) <= 1300


class TestSimpleTokenizerV1:
    def test_encode_returns_ids_for_known_tokens(self) -> None:
        vocab = build_vocab("Hello, world!")
        tokenizer = SimpleTokenizerV1(vocab)
        ids = tokenizer.encode("Hello, world!")
        assert ids == [vocab["Hello"], vocab[","], vocab["world"], vocab["!"]]

    def test_encode_raises_on_unknown_token(self) -> None:
        vocab = build_vocab("Hello, world!")
        tokenizer = SimpleTokenizerV1(vocab)
        with pytest.raises(KeyError):
            tokenizer.encode("Goodbye, world!")

    def test_decode_reconstructs_simple_text(self) -> None:
        text = "Hello, world!"
        vocab = build_vocab(text)
        tokenizer = SimpleTokenizerV1(vocab)
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_encode_decode_round_trip_on_verdict_text(
        self, verdict_text: str, verdict_vocab: dict[str, int]
    ) -> None:
        tokenizer = SimpleTokenizerV1(verdict_vocab)
        sample = verdict_text[:200]
        ids = tokenizer.encode(sample)
        decoded = tokenizer.decode(ids)
        # Not byte-identical (decode() always joins with single spaces and
        # doesn't special-case "--"), but every original word/punctuation
        # token must still appear, in order.
        assert tokenize(decoded) == tokenize(sample)

    def test_str_to_int_and_int_to_str_are_inverses(self, verdict_vocab: dict[str, int]) -> None:
        tokenizer = SimpleTokenizerV1(verdict_vocab)
        for token, idx in tokenizer.str_to_int.items():
            assert tokenizer.int_to_str[idx] == token


class TestSimpleTokenizerV2:
    def test_adds_special_tokens_to_vocab(self) -> None:
        vocab = build_vocab("Hello, world!")
        tokenizer = SimpleTokenizerV2(vocab)
        assert "<|unk|>" in tokenizer.str_to_int
        assert "<|endoftext|>" in tokenizer.str_to_int

    def test_does_not_duplicate_special_tokens_already_in_vocab(self) -> None:
        vocab = {"<|unk|>": 0, "<|endoftext|>": 1, "hello": 2}
        tokenizer = SimpleTokenizerV2(vocab)
        assert len(tokenizer.str_to_int) == 3

    def test_encode_maps_unknown_tokens_to_unk(self) -> None:
        vocab = build_vocab("Hello, world!")
        tokenizer = SimpleTokenizerV2(vocab)
        ids = tokenizer.encode("Goodbye, world!")
        unk_id = tokenizer.str_to_int["<|unk|>"]
        assert ids[0] == unk_id  # "Goodbye" is not in vocab
        assert ids[1] == vocab[","]
        assert ids[2] == vocab["world"]
        assert ids[3] == vocab["!"]

    def test_encode_never_raises_on_unknown_tokens(self) -> None:
        vocab = build_vocab("Hello, world!")
        tokenizer = SimpleTokenizerV2(vocab)
        # Should not raise, unlike SimpleTokenizerV1.
        tokenizer.encode("Completely unseen vocabulary appears here.")

    def test_encode_decode_round_trip_with_unknown_tokens(self) -> None:
        vocab = build_vocab("Hello, world!")
        tokenizer = SimpleTokenizerV2(vocab)
        text = "Hello, world! Goodbye, world!"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert "<|unk|>" in decoded

    def test_handles_endoftext_token_round_trip(self) -> None:
        vocab = build_vocab("Hello, world!")
        tokenizer = SimpleTokenizerV2(vocab)
        text = "Hello, world! <|endoftext|> Hello, world!"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert tokenize(decoded) == tokenize(text)

    def test_on_verdict_text_never_raises(
        self, verdict_text: str, verdict_vocab: dict[str, int]
    ) -> None:
        tokenizer = SimpleTokenizerV2(verdict_vocab)
        ids = tokenizer.encode(verdict_text + " some_completely_madeup_token_xyz")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
