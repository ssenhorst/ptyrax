import jax.numpy as jnp

from ptyrax.parametrizations import DirectArrayParametrization, IndexSliceParameter, resolve_index_dependent_parameters


class TestDirectArrayParametrization:
    def test_direct_initializer(self) -> None:
        data = jnp.reshape(jnp.arange(10 * 10 * 3), (10, 10, 3))
        model = DirectArrayParametrization(data)
        assert data.shape == model.output_shape
        assert jnp.all(data == model())

    def test_function_initializer(self) -> None:
        data = jnp.reshape(jnp.arange(10 * 10 * 3), (10, 10, 3))
        model = DirectArrayParametrization(data)
        assert data.shape == model.output_shape
        assert jnp.all(data == model())


class TestIndexSliceParameter:
    def test_slicing(self) -> None:
        data = jnp.reshape(jnp.arange(4 * 3 * 2), (4, 3, 2))
        model = IndexSliceParameter(data)
        for i in range(4):
            expected = data[i]
            actual = model.at_index(i)
            assert jnp.all(expected == actual)


def test_resolve_index_slice_parametrizations() -> None:
    data = jnp.reshape(jnp.arange(4 * 3 * 2), (4, 3, 2))
    model = IndexSliceParameter(data)
    resolved = resolve_index_dependent_parameters(model, 2)
    expected = data[2]
    assert jnp.all(expected == resolved.at_current_index())


def test_unresolved_throws_error() -> None:
    data = jnp.reshape(jnp.arange(4 * 3 * 2), (4, 3, 2))
    model = IndexSliceParameter(data)
    try:
        _ = model.at_current_index()
        assert False, "Expected an error when calling unresolved IndexSliceParameter"
    except ValueError as e:
        assert len(str(e)) > 0
