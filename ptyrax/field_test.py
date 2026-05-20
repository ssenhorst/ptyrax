import jax.numpy as jnp
import numpy as np

from ptyrax.field import CoherentField
from ptyrax.spatial import (
    CoordinateSystem,
    R_y,
    Rotation,
    SamplingGrid,
)


class TestCoherentField:
    def test_init(self) -> None:
        data = jnp.reshape(jnp.arange(10 * 10 * 10 * 3), (10, 10, 10, 3)) - 1j * jnp.reshape(
            jnp.arange(10 * 10 * 10 * 3), (10, 10, 10, 3)
        )
        sampling = SamplingGrid.from_tuples(shape=(10, 10, 10), pixel_size=(1.0, 1.0, 1.0))
        field = CoherentField(data, 1, sampling)
        assert field().shape == data.shape

    def test_amplitude(self) -> None:
        data = jnp.reshape(jnp.arange(10 * 10 * 10 * 3), (10, 10, 10, 3)) - 1j * jnp.reshape(
            jnp.arange(10 * 10 * 10 * 3), (10, 10, 10, 3)
        )
        sampling = SamplingGrid.from_tuples(shape=(10, 10, 10), pixel_size=(1.0, 1.0, 1.0))
        field = CoherentField(data, 1.0, sampling)
        assert jnp.all(field.amplitude >= 0)

    def test_propagation_direction(self) -> None:
        data = jnp.reshape(jnp.arange(10 * 10 * 10 * 3), (10, 10, 10, 3)) - 1j * jnp.reshape(
            jnp.arange(10 * 10 * 10 * 3), (10, 10, 10, 3)
        )
        sampling = SamplingGrid.from_tuples(shape=(10, 10, 10), pixel_size=(1.0, 1.0, 1.0))
        field = CoherentField(data, 1.0, sampling)
        assert field.propagation_direction.shape == (3,)

    def test_to_fftshifted(self) -> None:
        """
        Testing whether:
        1. to_fft_shifted and to_fft_unshifted are mutual inverses
        2. Whether both are idempotent operators
        """
        data = jnp.reshape(jnp.arange(10 * 10 * 10 * 3), (10, 10, 10, 3)) - 1j * jnp.reshape(
            jnp.arange(10 * 10 * 10 * 3), (10, 10, 10, 3)
        )
        sampling = SamplingGrid.from_tuples(shape=(10, 10, 10), pixel_size=(1.0, 1.0, 1.0))
        field = CoherentField(data, 1.0, sampling).to_fft_shifted
        np.testing.assert_allclose(field.to_fft_shifted.data, field.data)
        np.testing.assert_allclose(field.to_fft_unshifted.to_fft_shifted.data, field.data)
        np.testing.assert_allclose(field.to_fft_unshifted.to_fft_unshifted.data, field.to_fft_unshifted.data)

    def test_to_ifftshifted(self) -> None:
        data = jnp.reshape(jnp.arange(10 * 10 * 10 * 3), (10, 10, 10, 3)) - 1j * jnp.reshape(
            jnp.arange(10 * 10 * 10 * 3), (10, 10, 10, 3)
        )
        sampling = SamplingGrid.from_tuples(shape=(10, 10, 10), pixel_size=(1.0, 1.0, 1.0))
        field = CoherentField(data, 1.0, sampling).to_fft_shifted
        field_unshifted = field.to_fft_unshifted

        assert jnp.allclose(field_unshifted.to_fft_unshifted.data, field_unshifted.data)
        assert jnp.allclose(field_unshifted.to_fft_shifted.to_fft_unshifted.data, field_unshifted.data)

    def test_model_planewave_normal_incidence(self) -> None:
        data = jnp.ones((2, 2, 1), dtype=jnp.complex64)
        sampling = SamplingGrid.from_tuples(shape=data.shape[:2], pixel_size=(1.0, 1.0))
        coordinates = CoordinateSystem(
            translation=jnp.array([0.0, 0.0, 0.0]),
            rotation=Rotation.from_matrix(jnp.eye(3)),
        )
        propagation_direction = jnp.array([0.0, 0.0, 1.0])
        field = CoherentField(
            data,
            1.0,
            sampling,
            propagation_direction=propagation_direction,
            coordinate_system=coordinates,
        )
        pupil = field.propagate_fraunhofer(1.0)  # To angular space
        pupil_meshgrid = pupil.sampling.meshgrid
        np.testing.assert_almost_equal(pupil_meshgrid[0, 1, 1], 0.0, decimal=4)

    def test_model_tilted_planewave_global_coordinates_propagation(self) -> None:
        angle = jnp.deg2rad(30.0)
        displacement = jnp.array([0.0, 0.0, 0.5])
        wavelength = 1.0
        data = jnp.ones((10, 10, 1), dtype=jnp.complex64)
        sampling = SamplingGrid.from_tuples(shape=(10, 10), pixel_size=(1.0, 1.0))
        coordinates = CoordinateSystem(
            translation=jnp.array([0.0, 0.0, 0.0]),
            rotation=Rotation.from_matrix(jnp.eye(3)),
        )
        propagation_direction = jnp.array([jnp.sin(angle), 0.0, jnp.cos(angle)])
        field = CoherentField(
            data,
            1.0,
            sampling,
            propagation_direction=propagation_direction,
            coordinate_system=coordinates,
        )
        prop_field = field.propagate_tilted_nearfield(displacement=displacement)
        expected_phase = -(2 * jnp.pi / wavelength) * (displacement[2] * propagation_direction[2])
        np.testing.assert_almost_equal(np.mean(jnp.angle(prop_field())), expected_phase, decimal=4)

    def test_model_tilted_planewave_local_coordinate_propagation(self) -> None:
        angle = jnp.deg2rad(30.0)
        displacement = jnp.array([0.0, 0.0, 0.5])
        wavelength = 1.0
        data = jnp.ones((10, 10, 1), dtype=jnp.complex64)
        sampling = SamplingGrid.from_tuples(shape=(10, 10), pixel_size=(1.0, 1.0))
        coordinates = CoordinateSystem(
            translation=jnp.array([0.0, 0.0, 0.0]),
            rotation=Rotation.from_matrix(R_y(jnp.rad2deg(angle))),
        )
        propagation_direction = jnp.array([0.0, 0.0, 1.0])
        field = CoherentField(
            data,
            1.0,
            sampling,
            propagation_direction=propagation_direction,
            coordinate_system=coordinates,
        )
        prop_field = field.propagate_tilted_nearfield(displacement=displacement)
        k = 2 * jnp.pi / wavelength
        internal_propagation_direction = field.coordinate_system.rotation.as_matrix() @ propagation_direction
        expected_phase = -k * (displacement @ (internal_propagation_direction))
        np.testing.assert_almost_equal(np.mean(jnp.angle(prop_field())), expected_phase, decimal=4)

    def test_k_z_matches_expected_dispersion_relation(self) -> None:
        wavelength = 1.0
        sampling = SamplingGrid.from_tuples(shape=(8, 8), pixel_size=(1.0, 1.0))
        field = CoherentField(jnp.ones((8, 8, 1), dtype=jnp.complex64), wavelength, sampling)

        k_z, valid = field.k_z()

        far_field = sampling.to_far_field(wavelength=wavelength, propagation_distance=1.0, fftshifted=False)
        xi_x, xi_y = far_field.meshgrid
        radial_sq = xi_x**2 + xi_y**2
        expected_valid = radial_sq <= (1.0 / wavelength) ** 2
        expected_k_z = jnp.sqrt(jnp.maximum((1.0 / wavelength) ** 2 - radial_sq, 0.0))

        np.testing.assert_array_equal(np.asarray(valid), np.asarray(expected_valid))
        np.testing.assert_allclose(np.asarray(k_z), np.asarray(expected_k_z), rtol=1e-6, atol=1e-6)

    def test_multiply_fourier_zero_factor(self) -> None:
        sampling = SamplingGrid.from_tuples(shape=(8, 8), pixel_size=(1.0, 1.0))
        data = jnp.reshape(jnp.arange(64), (8, 8, 1)).astype(jnp.float32) + 1j * jnp.reshape(
            jnp.arange(64), (8, 8, 1)
        ).astype(jnp.float32)
        field = CoherentField(data, 1.0, sampling)

        multiplied = field.multiply_fourier(jnp.zeros((8, 8), dtype=jnp.complex64))
        np.testing.assert_allclose(np.asarray(multiplied()), np.zeros_like(np.asarray(field())), rtol=1e-6, atol=1e-6)

    def test_multiply_real_scales_pointwise(self) -> None:
        sampling = SamplingGrid.from_tuples(shape=(4, 4), pixel_size=(1.0, 1.0))
        data = jnp.ones((4, 4, 1), dtype=jnp.complex64) * (2.0 + 3.0j)
        factor = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
        field = CoherentField(data, 1.0, sampling)

        multiplied = field.multiply_real(factor)
        expected = data * factor[..., None]
        np.testing.assert_allclose(np.asarray(multiplied()), np.asarray(expected), rtol=1e-6, atol=1e-6)

    def test_add_returns_data_sum(self) -> None:
        sampling = SamplingGrid.from_tuples(shape=(4, 4), pixel_size=(1.0, 1.0))
        field_a = CoherentField(jnp.ones((4, 4, 1), dtype=jnp.complex64), 1.0, sampling)
        field_b = CoherentField(jnp.ones((4, 4, 1), dtype=jnp.complex64) * (2.0 - 1.0j), 1.0, sampling)

        summed = field_a + field_b
        np.testing.assert_allclose(np.asarray(summed()), np.asarray(field_a() + field_b()), rtol=1e-6, atol=1e-6)


def test_import_field() -> None:
    __import__("ptyrax.field")
