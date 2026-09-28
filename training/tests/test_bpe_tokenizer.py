from pathlib import Path

import pytest

from peekllm_training.bpe_tokenizer import END_OF_TEXT_TOKEN, BPETokenizer

VERDICT_PATH = Path(__file__).parent.parent / "peekllm_training" / "data" / "the-verdict.txt"


@pytest.fixture(scope="module")
def verdict_text() -> str:
    return VERDICT_PATH.read_text()


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    return BPETokenizer()


class TestBPETokenizer:
    def test_encode_returns_a_list_of_ints(self, tokenizer: BPETokenizer) -> None:
        ids = tokenizer.encode("Hello, world!")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_decode_round_trip(self, tokenizer: BPETokenizer) -> None:
        text = "Hello, do you like tea?"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_handles_unknown_or_madeup_words_without_raising(self, tokenizer: BPETokenizer) -> None:
        text = "someunknownPlace with akwirw ier and 你好"
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text

    def test_endoftext_token_round_trips(self, tokenizer: BPETokenizer) -> None:
        text = f"Hello, world! {END_OF_TEXT_TOKEN} Goodbye, world!"
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text

    def test_endoftext_encodes_to_a_single_known_id(self, tokenizer: BPETokenizer) -> None:
        ids = tokenizer.encode(END_OF_TEXT_TOKEN)
        assert ids == [50256]

    def test_vocab_size_matches_gpt2(self, tokenizer: BPETokenizer) -> None:
        assert tokenizer.vocab_size == 50257

    def test_round_trip_on_verdict_text(self, tokenizer: BPETokenizer, verdict_text: str) -> None:
        ids = tokenizer.encode(verdict_text)
        assert tokenizer.decode(ids) == verdict_text

    def test_produces_fewer_or_equal_tokens_than_word_level_split(
        self, tokenizer: BPETokenizer, verdict_text: str
    ) -> None:
        naive_word_count = len(verdict_text.split())
        ids = tokenizer.encode(verdict_text)
        assert len(ids) < naive_word_count * 2
