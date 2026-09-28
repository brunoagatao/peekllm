from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class EmbeddingParams:
    """The two embedding tables a GPT-style model looks up before layer 1.

    `token_embedding` has shape (vocab_size, embedding_dim): one learnable
    row per token id. `position_embedding` has shape
    (context_length, embedding_dim): one learnable row per sequence
    position, up to the maximum sequence length the model supports.
    """

    token_embedding: jax.Array
    position_embedding: jax.Array

    @property
    def vocab_size(self) -> int:
        return self.token_embedding.shape[0]

    @property
    def context_length(self) -> int:
        return self.position_embedding.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.token_embedding.shape[1]

    @classmethod
    def init(
        cls,
        key: jax.Array,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
    ) -> EmbeddingParams:
        """Initialize both tables from a single PRNG key.

        Matches `nn.Embedding`'s default initialization: entries drawn
        independently from a standard normal distribution. The token and
        position tables get their own subkey each (via `jax.random.split`)
        so they aren't accidentally initialized to the same values.
        """
        token_key, position_key = jax.random.split(key)
        return cls(
            token_embedding=jax.random.normal(token_key, (vocab_size, embedding_dim)),
            position_embedding=jax.random.normal(position_key, (context_length, embedding_dim)),
        )


def embed(params: EmbeddingParams, token_ids: jax.Array) -> jax.Array:
    """Look up token + positional embeddings and combine them.

    `token_ids` has shape (batch_size, seq_len); the result has shape
    (batch_size, seq_len, embedding_dim). The same positional embedding row
    is added to every sequence in the batch at a given position (broadcast
    over the batch dimension).
    """
    _batch_size, seq_len = token_ids.shape
    if seq_len > params.context_length:
        raise ValueError(f"seq_len ({seq_len}) exceeds context_length ({params.context_length})")

    token_embeddings = params.token_embedding[token_ids]
    position_embeddings = params.position_embedding[jnp.arange(seq_len)]
    return token_embeddings + position_embeddings
