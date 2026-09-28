import jax
import jax.numpy as jnp
import pytest

from peekllm_training.embeddings import EmbeddingParams, embed

VOCAB_SIZE = 6
CONTEXT_LENGTH = 4
EMBEDDING_DIM = 3


@pytest.fixture
def params() -> EmbeddingParams:
    key = jax.random.key(0)
    return EmbeddingParams.init(key, VOCAB_SIZE, CONTEXT_LENGTH, EMBEDDING_DIM)


class TestEmbeddingParamsInit:
    def test_table_shapes(self, params: EmbeddingParams) -> None:
        assert params.token_embedding.shape == (VOCAB_SIZE, EMBEDDING_DIM)
        assert params.position_embedding.shape == (CONTEXT_LENGTH, EMBEDDING_DIM)

    def test_exposes_size_properties(self, params: EmbeddingParams) -> None:
        assert params.vocab_size == VOCAB_SIZE
        assert params.context_length == CONTEXT_LENGTH
        assert params.embedding_dim == EMBEDDING_DIM

    def test_deterministic_given_same_key(self) -> None:
        key = jax.random.key(42)
        first = EmbeddingParams.init(key, VOCAB_SIZE, CONTEXT_LENGTH, EMBEDDING_DIM)
        second = EmbeddingParams.init(key, VOCAB_SIZE, CONTEXT_LENGTH, EMBEDDING_DIM)
        assert jnp.array_equal(first.token_embedding, second.token_embedding)
        assert jnp.array_equal(first.position_embedding, second.position_embedding)

    def test_different_keys_give_different_tables(self) -> None:
        first = EmbeddingParams.init(jax.random.key(0), VOCAB_SIZE, CONTEXT_LENGTH, EMBEDDING_DIM)
        second = EmbeddingParams.init(jax.random.key(1), VOCAB_SIZE, CONTEXT_LENGTH, EMBEDDING_DIM)
        assert not jnp.array_equal(first.token_embedding, second.token_embedding)

    def test_token_and_position_tables_are_not_identical(self, params: EmbeddingParams) -> None:
        same_size_params = EmbeddingParams.init(
            jax.random.key(0), embedding_dim=EMBEDDING_DIM, vocab_size=5, context_length=5
        )
        assert not jnp.array_equal(
            same_size_params.token_embedding, same_size_params.position_embedding
        )


class TestEmbed:
    def test_output_shape(self, params: EmbeddingParams) -> None:
        token_ids = jnp.array([[0, 1, 2], [3, 4, 5]])
        result = embed(params, token_ids)
        assert result.shape == (2, 3, EMBEDDING_DIM)

    def test_equals_token_embedding_plus_position_embedding(self, params: EmbeddingParams) -> None:
        token_ids = jnp.array([[2, 0, 5]])
        result = embed(params, token_ids)
        expected = params.token_embedding[token_ids] + params.position_embedding[jnp.arange(3)]
        assert jnp.array_equal(result, expected)

    def test_same_token_and_position_gives_identical_output_across_batch(
        self, params: EmbeddingParams
    ) -> None:
        token_ids = jnp.array([[0, 1], [2, 1]])
        result = embed(params, token_ids)
        assert jnp.array_equal(result[0, 1], result[1, 1])

    def test_repeated_token_id_gets_identical_token_embedding(
        self, params: EmbeddingParams
    ) -> None:
        token_ids = jnp.array([[3, 3]])
        result = embed(params, token_ids)
        position_delta = params.position_embedding[1] - params.position_embedding[0]
        assert jnp.allclose(result[0, 1] - result[0, 0], position_delta)

    def test_raises_when_seq_len_exceeds_context_length(self, params: EmbeddingParams) -> None:
        too_long = jnp.zeros((1, CONTEXT_LENGTH + 1), dtype=jnp.int32)
        with pytest.raises(ValueError, match="exceeds context_length"):
            embed(params, too_long)

    def test_allows_seq_len_equal_to_context_length(self, params: EmbeddingParams) -> None:
        exact = jnp.zeros((1, CONTEXT_LENGTH), dtype=jnp.int32)
        result = embed(params, exact)
        assert result.shape == (1, CONTEXT_LENGTH, EMBEDDING_DIM)
