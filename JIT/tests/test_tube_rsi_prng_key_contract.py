from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jit_dvgc.tube_rsi import _match_prng_key_representation


def test_tube_rng_representation_matches_typed_and_legacy_callers_without_changing_key_data():
    stored_data = jnp.asarray([123, 456], dtype=jnp.uint32)
    stored_key = jax.random.wrap_key_data(stored_data)

    typed_reference = jax.random.key(7)
    typed = _match_prng_key_representation(stored_key, typed_reference)
    assert jax.dtypes.issubdtype(typed.dtype, jax.dtypes.prng_key)
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(typed)), np.asarray(stored_data)
    )
    selected_typed = jnp.where(jnp.asarray(True), typed, typed_reference)
    assert jax.dtypes.issubdtype(selected_typed.dtype, jax.dtypes.prng_key)

    legacy_reference = jax.random.PRNGKey(7)
    legacy = _match_prng_key_representation(stored_key, legacy_reference)
    assert legacy.dtype == jnp.uint32
    np.testing.assert_array_equal(np.asarray(legacy), np.asarray(stored_data))
    selected_legacy = jnp.where(jnp.asarray(True), legacy, legacy_reference)
    assert selected_legacy.dtype == jnp.uint32
