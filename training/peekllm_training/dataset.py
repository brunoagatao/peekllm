from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

import jax.numpy as jnp
import numpy as np


class Tokenizer(Protocol):
    """Structural type for anything with an `encode(text) -> list[int]` method.

    Satisfied by SimpleTokenizerV1/V2 and BPETokenizer without either of them
    needing to know about this module.
    """

    def encode(self, text: str) -> list[int]: ...


@dataclass
class GPTDataset:
    """Every (input, target) sliding-window pair for a piece of text.

    Each input chunk is `max_length` consecutive token IDs; its target chunk
    is the same window shifted one token to the right, i.e. "predict the
    next token" at every position in the chunk. `stride` controls how far
    the window advances between chunks: `stride == max_length` gives
    non-overlapping chunks, `stride < max_length` gives overlapping ones
    (more training examples from the same text, at the cost of redundancy).
    """

    input_ids: np.ndarray  # shape: (num_windows, max_length)
    target_ids: np.ndarray  # shape: (num_windows, max_length)

    @classmethod
    def from_text(cls, text: str, tokenizer: Tokenizer, max_length: int, stride: int) -> GPTDataset:
        token_ids = tokenizer.encode(text)

        inputs = []
        targets = []
        for start in range(0, len(token_ids) - max_length, stride):
            inputs.append(token_ids[start : start + max_length])
            targets.append(token_ids[start + 1 : start + max_length + 1])

        return cls(
            input_ids=np.array(inputs, dtype=np.int64),
            target_ids=np.array(targets, dtype=np.int64),
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.input_ids[idx], self.target_ids[idx]


@dataclass
class DataLoader:
    """Iterates over a GPTDataset in batches of JAX arrays.

    Re-shuffles on every fresh `iter(...)` call when `shuffle=True` (i.e.
    every `for batch in loader:` loop), so the same DataLoader can be reused
    across training epochs with a different order each time.
    """

    dataset: GPTDataset
    batch_size: int = 4
    shuffle: bool = True
    drop_last: bool = True
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        num_batches, remainder = divmod(len(self.dataset), self.batch_size)
        if not self.drop_last and remainder:
            num_batches += 1
        return num_batches

    def __iter__(self) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            yield (
                jnp.asarray(self.dataset.input_ids[batch_indices]),
                jnp.asarray(self.dataset.target_ids[batch_indices]),
            )


def create_dataloader(
    text: str,
    tokenizer: Tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int | None = None,
) -> DataLoader:
    """Build a GPTDataset from raw text and wrap it in a DataLoader."""
    dataset = GPTDataset.from_text(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )
