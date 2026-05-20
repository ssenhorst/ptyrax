import equinox as eqx
import jax.numpy as jnp
import jax.tree as tree
import numpy as np
from jaxtyping import Array, Complex

from ptyrax.field import CoherentField
from ptyrax.parametrizations import resolve_parametrizations
from ptyrax.spatial import (
    CoordinateSystem,
    R_y,
    Rotation,
    SamplingGrid,
    matrix_to_six_dimensional_representation,
)


def test_interpolate_grid_to_grid() -> None:
    from ptyrax.spatial import SamplingGrid, interpolate_grid_to_grid

    grid = SamplingGrid.from_tuples(shape=(15, 15), pixel_size=(1.0, 1.0))
    sampled_coordinate_system = CoordinateSystem(translation=jnp.array([0.0, 2.0, 0.0]), rotation=Rotation())
    new_grid = SamplingGrid.from_tuples(shape=(20, 20), pixel_size=(0.5, 0.5))
    new_coordinate_system = CoordinateSystem(translation=jnp.array([0.0, 0.0, 0.0]), rotation=Rotation())
    data = jnp.reshape(jnp.arange(15 * 15), (15, 15))
    interpolated = interpolate_grid_to_grid(data, grid, sampled_coordinate_system, new_grid, new_coordinate_system)
    assert jnp.all(jnp.asarray(interpolated.shape) == jnp.asarray(new_grid.shape))


def test_interpolate_grid_to_grid_complex() -> None:
    from ptyrax.spatial import SamplingGrid, interpolate_grid_to_grid

    grid = SamplingGrid.from_tuples(shape=(15, 15), pixel_size=(1.0, 1.0))
    sampled_coordinate_system = CoordinateSystem(translation=jnp.array([0.0, 2.0, 0.0]), rotation=Rotation())
    new_grid = SamplingGrid.from_tuples(shape=(20, 20), pixel_size=(0.5, 0.5))
    new_coordinate_system = CoordinateSystem(translation=jnp.array([0.0, 0.0, 0.0]), rotation=Rotation())
    data = jnp.reshape(jnp.arange(15 * 15), (15, 15)) * (1 + 1j)
    interpolated = interpolate_grid_to_grid(data, grid, sampled_coordinate_system, new_grid, new_coordinate_system)
    assert jnp.all(jnp.asarray(interpolated.shape) == jnp.asarray(new_grid.shape))


class TestCoordinateSystem:
    def test_axis(self) -> None:
        system = CoordinateSystem()
        assert jnp.all(system.x_axis == jnp.array([1.0, 0.0, 0.0]))
        assert system.x_axis.shape == (3,)


class TestRotation:
    def test_matrix_shape(self) -> None:
        rotation = Rotation()
        assert rotation.as_matrix().shape == (3, 3)


class TestYOnlyRotation:
    def test_matrix_shape(self) -> None:
        rotation = Rotation()
        eqx.tree_at(lambda t: tree.leaves(t), rotation, replace_fn=lambda x: x + 10)
        assert rotation.as_matrix().shape == (3, 3)


def test_shift_with_interpolation_pixel_size() -> None:
    from ptyrax.spatial import shift_with_interpolation_unequal_pixel_size

    a = np.arange(100 * 100.0).reshape((1, 100, 100))
    a_pixel_size = np.array((2.0, 2.0))
    shift = np.array((0.0, 0.0))
    b_shape = np.array((10, 10))
    b_pixel_size = np.array((20.0, 20.0))
    interpolated = shift_with_interpolation_unequal_pixel_size(
        a,
        a_pixel_size,
        shift,
        b_shape,
        b_pixel_size,
    )
    assert np.allclose(interpolated, a[:, ::10, ::10])


def test_shift_with_interpolation_pixel_size_center() -> None:
    from ptyrax.spatial import shift_with_interpolation_unequal_pixel_size

    a = np.arange(100 * 100).reshape((1, 100, 100))
    a_pixel_size = np.array((2.0, 2.0))
    shift = np.array((-25.0, -25.0))
    b_shape = np.array((10, 10))
    b_pixel_size = np.array((10.0, 10.0))
    interpolated = shift_with_interpolation_unequal_pixel_size(
        a,
        a_pixel_size,
        shift,
        b_shape,
        b_pixel_size,
    )
    assert np.allclose(interpolated, a[:, :50:5, :50:5])


def test_six_dimensional_representation_inverse() -> None:
    test_angle = np.deg2rad(45.0)
    test_matrix = np.array(
        [[np.cos(test_angle), -np.sin(test_angle), 0.0], [np.sin(test_angle), np.cos(test_angle), 0.0], [0.0, 0.0, 1.0]]
    )

    from ptyrax.spatial import six_dimensional_representation_to_matrix

    representation = matrix_to_six_dimensional_representation(test_matrix)
    inverse = six_dimensional_representation_to_matrix(representation)
    assert np.allclose(test_matrix, inverse)


class TestFresnelReflection2D:
    def test_zero_illumination(self) -> None:
        from ptyrax.models.ptychography import FresnelReflection

        def data_initializer(sampling: SamplingGrid, *args) -> Complex[Array, "height width"]:
            shape = sampling.shape
            return jnp.reshape(jnp.arange(shape[0] * 1), shape=(shape[0], 1)) - 1j * jnp.reshape(
                jnp.arange(1 * shape[1]), (1, shape[1])
            )

        grid = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(2.0, 2.0))
        coordinates = CoordinateSystem(
            translation=jnp.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]),
            rotation=Rotation.from_matrix(jnp.tile(jnp.eye(3), (2, 1, 1))),
        )
        fresnel = FresnelReflection(coordinates, grid, initializer=data_initializer)
        illumination_field = CoherentField(
            jnp.zeros((100, 100, 1), dtype=jnp.complex64),
            jnp.array(1.0),
            SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0)),
        )
        output_field = resolve_parametrizations(fresnel, index=0)(illumination_field)
        assert jnp.all(output_field() == jnp.zeros((100, 100), dtype=jnp.complex64))

    def test_illumination_single_zero(self) -> None:
        from ptyrax.models.ptychography import FresnelReflection

        def data_initializer(sampling: SamplingGrid, *args) -> Complex[Array, "height width"]:
            shape = sampling.shape
            return jnp.reshape(jnp.arange(shape[0] * 1), shape=(shape[0], 1)) - 1j * jnp.reshape(
                jnp.arange(1 * shape[1]), (1, shape[1])
            )

        grid = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(2.0, 2.0))
        coordinates = CoordinateSystem(
            translation=jnp.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]),
            rotation=Rotation.from_matrix(jnp.tile(jnp.eye(3), (2, 1, 1))),
        )
        fresnel = FresnelReflection(coordinates, grid, initializer=data_initializer)
        data = np.ones((100, 100, 1), dtype=jnp.complex64)
        data[5, 5] = 0.0
        illumination_field = CoherentField(
            jnp.asarray(data),
            1,
            SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0)),
        )
        output_field = resolve_parametrizations(fresnel, index=0)(illumination_field)
        assert output_field()[5, 5] == 0

    def test_coordinates_without_shift(self) -> None:
        from ptyrax.models.ptychography import FresnelReflection

        def data_initializer(sampling: SamplingGrid, *args) -> Complex[Array, "height width"]:
            shape = sampling.shape
            grid = SamplingGrid.from_tuples(shape=shape, pixel_size=(1.0, 1.0))
            return grid.meshgrid[0] + 1j * grid.meshgrid[1]

        grid = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0))
        coordinates = CoordinateSystem(
            translation=jnp.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            rotation=Rotation.from_matrix(jnp.tile(jnp.eye(3), (2, 1, 1))),
        )
        fresnel = FresnelReflection(coordinates, grid, initializer=data_initializer)
        illumination_data = np.ones((100, 100, 1), dtype=jnp.complex64)
        illumination_grid = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0))
        illumination_field = CoherentField(jnp.asarray(illumination_data), 1, illumination_grid)
        post_interaction_field = resolve_parametrizations(fresnel, index=0)(illumination_field, normalize=False)()
        compare_to = illumination_grid.meshgrid[0] + 1j * illumination_grid.meshgrid[1]

        np.testing.assert_allclose(np.real(post_interaction_field)[..., 0], np.real(compare_to), atol=0.1)
        np.testing.assert_allclose(np.imag(post_interaction_field)[..., 0], np.imag(compare_to), atol=0.1)

    def test_coordinates_shift(self) -> None:
        from ptyrax.models.ptychography import FresnelReflection

        def data_initializer(sampling: SamplingGrid, *args) -> Complex[Array, "height width"]:
            shape = sampling.shape
            grid = SamplingGrid.from_tuples(shape=shape, pixel_size=(2.0, 2.0))
            return grid.meshgrid[0] + 1j * grid.meshgrid[1]

        grid = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(2.0, 2.0))
        coordinates = CoordinateSystem(
            translation=jnp.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            rotation=Rotation.from_matrix(jnp.tile(jnp.eye(3), (2, 1, 1))),
        )
        fresnel = FresnelReflection(coordinates, grid, initializer=data_initializer)
        illumination_data = np.ones((50, 50, 1), dtype=jnp.complex64)
        illumination_grid = SamplingGrid.from_tuples(shape=(50, 50), pixel_size=(2.0, 2.0))
        illumination_field = CoherentField(jnp.asarray(illumination_data), 1, illumination_grid)
        output_pos_1 = resolve_parametrizations(fresnel, index=1)(illumination_field, normalize=False)()
        output_pos_2 = resolve_parametrizations(fresnel, index=2)(illumination_field, normalize=False)()
        np.testing.assert_allclose(np.real(output_pos_1)[..., 0], illumination_grid.meshgrid[0], atol=1)
        np.testing.assert_allclose(np.imag(output_pos_1)[..., 0], illumination_grid.meshgrid[1] + 1.0, atol=1)
        np.testing.assert_allclose(np.real(output_pos_2)[..., 0], illumination_grid.meshgrid[0] + 1.0, atol=1)
        np.testing.assert_allclose(np.imag(output_pos_2)[..., 0], illumination_grid.meshgrid[1], atol=1)

    def test_propagation_direction_uses_global_reflection_convention(self) -> None:
        from ptyrax.models.ptychography import FresnelReflection

        def data_initializer(sampling: SamplingGrid, *args) -> Complex[Array, "height width"]:
            return jnp.ones(sampling.shape, dtype=jnp.complex64)

        grid = SamplingGrid.from_tuples(shape=(16, 16), pixel_size=(1.0, 1.0))
        tilt = jnp.deg2rad(30.0)
        sample_rotation = Rotation.from_matrix(R_y(tilt)[None, ...])
        coordinates = CoordinateSystem(
            translation=jnp.array([[0.0, 0.0, 0.0]]),
            rotation=sample_rotation,
        )
        fresnel = FresnelReflection(coordinates, grid, initializer=data_initializer)

        incident = jnp.array([0.25, 0.0, jnp.sqrt(1.0 - 0.25**2)])
        illumination_field = CoherentField(
            jnp.ones((16, 16, 1), dtype=jnp.complex64),
            jnp.array(1.0),
            grid,
            propagation_direction=incident,
        )

        output_field = resolve_parametrizations(fresnel, index=0)(illumination_field, normalize=False)

        sample_z_axis = sample_rotation.as_matrix()[0].T @ jnp.array([0.0, 0.0, 1.0])
        expected = incident - 2.0 * jnp.dot(incident, sample_z_axis) * sample_z_axis

        np.testing.assert_allclose(output_field.propagation_direction, expected, rtol=1e-6, atol=1e-6)
