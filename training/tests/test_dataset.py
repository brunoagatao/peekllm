from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from peekllm_training.bpe_tokenizer import BPETokenizer
from peekllm_training.dataset import DataLoader, GPTDataset, create_dataloader

VERDICT_PATH = Path(__file__).parent.parent / "peekllm_training" / "data" / "the-verdict.txt"


class _WordTokenizer:
    """Minimal stand-in tokenizer: one token id per character, for tests
    that want easy-to-predict token ids without pulling in BPE/vocab setup.
    """

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]


@pytest.fixture(scope="module")
def verdict_text() -> str:
    return VERDICT_PATH.read_text()


@pytest.fixture(scope="module")
def bpe_tokenizer() -> BPETokenizer:
    return BPETokenizer()


class TestGPTDatasetFromText:
    def test_target_is_input_shifted_by_one_token(self) -> None:
        dataset = GPTDataset.from_text("abcdefgh", _WordTokenizer(), max_length=4, stride=1)
        input_ids, target_ids = dataset[0]
        np.testing.assert_array_equal(target_ids[:-1], input_ids[1:])

    def test_non_overlapping_windows_when_stride_equals_max_length(self) -> None:
        dataset = GPTDataset.from_text("abcdefgh", _WordTokenizer(), max_length=4, stride=4)
        assert len(dataset) == 1  # only one full non-overlapping window fits
        input_ids, _ = dataset[0]
        assert [chr(c) for c in input_ids] == ["a", "b", "c", "d"]

    def test_overlapping_windows_when_stride_less_than_max_length(self) -> None:
        dataset = GPTDataset.from_text("abcdefgh", _WordTokenizer(), max_length=4, stride=1)
        first_input, _ = dataset[0]
        second_input, _ = dataset[1]
        # Windows overlap by max_length - stride = 3 tokens.
        np.testing.assert_array_equal(first_input[1:], second_input[:-1])

    def test_drops_trailing_text_shorter_than_a_full_window(self) -> None:
        # 9 tokens, max_length=4, stride=4 -> windows start at 0 and 4; the
        # trailing 1 token isn't enough for another full window.
        dataset = GPTDataset.from_text("abcdefghi", _WordTokenizer(), max_length=4, stride=4)
        assert len(dataset) == 2

    def test_matches_book_reference_output_on_verdict_text(
        self, verdict_text: str, bpe_tokenizer: BPETokenizer
    ) -> None:
        dataset = GPTDataset.from_text(verdict_text, bpe_tokenizer, max_length=4, stride=1)
        input_ids, target_ids = dataset[0]
        assert input_ids.tolist() == [40, 367, 2885, 1464]
        assert target_ids.tolist() == [367, 2885, 1464, 1807]


class TestDataLoader:
    def test_batches_have_the_requested_shape(self) -> None:
        dataset = GPTDataset.from_text("abcdefghijklmnop", _WordTokenizer(), max_length=4, stride=4)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
        inputs, targets = next(iter(loader))
        assert inputs.shape == (2, 4)
        assert targets.shape == (2, 4)

    def test_batches_are_jax_arrays(self) -> None:
        dataset = GPTDataset.from_text("abcdefghijklmnop", _WordTokenizer(), max_length=4, stride=4)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
        inputs, _targets = next(iter(loader))
        assert isinstance(inputs, jnp.ndarray)

    def test_drop_last_discards_a_partial_final_batch(self) -> None:
        # 3 windows, batch_size=2 -> one full batch of 2, one partial of 1.
        dataset = GPTDataset.from_text("abcdefghijklmnop", _WordTokenizer(), max_length=4, stride=4)
        assert len(dataset) == 3
        loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)
        batches = list(loader)
        assert len(batches) == 1
        assert len(loader) == 1

    def test_keeps_partial_final_batch_when_drop_last_is_false(self) -> None:
        dataset = GPTDataset.from_text("abcdefghijklmnop", _WordTokenizer(), max_length=4, stride=4)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
        batches = list(loader)
        assert len(batches) == 2
        assert batches[-1][0].shape == (1, 4)
        assert len(loader) == 2

    def test_without_shuffle_preserves_dataset_order(self) -> None:
        dataset = GPTDataset.from_text("abcdefghijklmnop", _WordTokenizer(), max_length=4, stride=4)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
        first_batch_input, _ = next(iter(loader))
        np.testing.assert_array_equal(np.asarray(first_batch_input[0]), dataset[0][0])

    def test_shuffle_changes_order_but_not_contents(self) -> None:
        dataset = GPTDataset.from_text(
            "abcdefghijklmnopqrstuvwxyz012345",
            _WordTokenizer(),
            max_length=4,
            stride=4,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False, seed=0)
        shuffled_inputs = {tuple(batch[0][0].tolist()) for batch in loader}
        original_inputs = {tuple(dataset[i][0].tolist()) for i in range(len(dataset))}
        # Same set of windows, just (very likely) a different draw order.
        assert shuffled_inputs == original_inputs

    def test_reshuffles_on_each_new_iteration(self) -> None:
        dataset = GPTDataset.from_text(
            "abcdefghijklmnopqrstuvwxyz0123456789",
            _WordTokenizer(),
            max_length=4,
            stride=4,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False, seed=0)
        first_pass = [tuple(batch[0][0].tolist()) for batch in loader]
        second_pass = [tuple(batch[0][0].tolist()) for batch in loader]
        assert first_pass != second_pass

    def test_create_dataloader_matches_book_reference_batch(
        self, verdict_text: str, bpe_tokenizer: BPETokenizer
    ) -> None:
        loader = create_dataloader(
            verdict_text,
            bpe_tokenizer,
            batch_size=8,
            max_length=4,
            stride=4,
            shuffle=False,
            drop_last=False,
        )
        inputs, targets = next(iter(loader))
        assert inputs.tolist() == [
            [40, 367, 2885, 1464],
            [1807, 3619, 402, 271],
            [10899, 2138, 257, 7026],
            [15632, 438, 2016, 257],
            [922, 5891, 1576, 438],
            [568, 340, 373, 645],
            [1049, 5975, 284, 502],
            [284, 3285, 326, 11],
        ]
        assert targets.tolist() == [
            [367, 2885, 1464, 1807],
            [3619, 402, 271, 10899],
            [2138, 257, 7026, 15632],
            [438, 2016, 257, 922],
            [5891, 1576, 438, 568],
            [340, 373, 645, 1049],
            [5975, 284, 502, 284],
            [3285, 326, 11, 287],
        ]
