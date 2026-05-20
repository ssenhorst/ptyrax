import jax
import jax.numpy as jnp

from ptyrax.models.propagation import (
    CoherentField,
    FarfieldPropagator,
    nearfield_propagation_coefficient_fourier,
)
from ptyrax.spatial import (
    CoordinateSystem,
    SamplingGrid,
)


def test_import_models_propagation() -> None:
    __import__("ptyrax.models.propagation")


class TestFarFieldPropagator:
    def test_uniform(self) -> None:
        propagator = FarfieldPropagator()
        data = jnp.reshape(jnp.arange(10 * 10 * 1), (10, 10, 1)) - 1j * jnp.reshape(
            jnp.arange(10 * 10 * 1), (10, 10, 1)
        )
        sampling = SamplingGrid.from_tuples(shape=(10, 10), pixel_size=jnp.array([1.0, 1.0]))
        input_field = jax.vmap(CoherentField, in_axes=(None, 0, None))(data, jnp.array([1.0]), sampling)

        output_sampling = SamplingGrid.from_tuples(shape=(10, 10), pixel_size=jnp.array((100.0, 100.0)))
        output_coordinates = CoordinateSystem(
            rotation=input_field.coordinate_system.rotation,
            translation=input_field.coordinate_system.translation + jnp.array([0.0, 0.0, 1000.0]),
        )

        output_field, mask = jax.vmap(propagator, in_axes=(0, 0, None))(
            input_field, output_coordinates, output_sampling
        )
        assert output_field().shape == input_field().shape


def test_nearfield_propagation_coefficient_fourier_with_valid_mask() -> None:
    k_z = jnp.array([[0.0, 0.5], [1.0, 1.5]])
    valid = jnp.array([[True, False], [True, False]])
    z_distance = 0.25

    coefficient = nearfield_propagation_coefficient_fourier(k_z=k_z, z_distance=z_distance, valid=valid)
    expected = jnp.exp(-1j * 2 * jnp.pi * k_z * z_distance)
    expected = jnp.where(valid, expected, 0.0)

    assert jnp.allclose(coefficient, expected)


def test_nearfield_propagation_coefficient_fourier_without_mask() -> None:
    k_z = jnp.array([[0.0, 0.5], [1.0, 1.5]])
    z_distance = 0.25

    coefficient = nearfield_propagation_coefficient_fourier(k_z=k_z, z_distance=z_distance)
    expected = jnp.exp(-1j * 2 * jnp.pi * k_z * z_distance)

    assert jnp.allclose(coefficient, expected)
