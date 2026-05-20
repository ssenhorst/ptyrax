import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex

from ptyrax.models.interaction import (
    CoherentField,
)
from ptyrax.parametrizations import resolve_parametrizations
from ptyrax.spatial import (
    CoordinateSystem,
    R_y,
    Rotation,
    SamplingGrid,
)


def test_import_models_interaction() -> None:
    __import__("ptyrax.models.interaction")


class TestMultiSlice:
    def test_basic_init(self) -> None:
        from ptyrax.models.interaction import MultiSlice

        sampling = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0))
        coordinates = CoordinateSystem(translation=jnp.array([0.0, 0.0, 0.0]), rotation=Rotation())

        def data_initializer(shape: tuple[int, int], *args) -> Complex[Array, "height width"]:
            return jnp.reshape(jnp.arange(shape[0] * shape[1]), shape=shape) - 1j * jnp.reshape(
                jnp.arange(shape[0] * shape[1]), shape=shape
            )

        multislice = MultiSlice(coordinates, sampling, jnp.array([0.0, 1.0]))
        assert multislice.slice_distances.shape[0] == 2

    def test_init_inner_interactions(self) -> None:
        from ptyrax.models.interaction import FresnelReflection, MultiSlice

        sampling = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0))
        coordinates = CoordinateSystem(
            translation=jnp.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]),
            rotation=Rotation.from_matrix(jnp.tile(jnp.eye(3), (2, 1, 1))),
        )

        def data_initializer(sampling: SamplingGrid, *args) -> Complex[Array, "height width"]:
            shape = sampling.shape
            return jnp.reshape(jnp.arange(shape[0] * 1), shape=(shape[0], 1)) - 1j * jnp.reshape(
                jnp.arange(1 * shape[1]), (1, shape[1])
            )

        interaction_1 = FresnelReflection(coordinates, sampling, initializer=data_initializer)
        interaction_2 = FresnelReflection(coordinates, sampling, initializer=data_initializer)

        multislice = MultiSlice.from_interactions(
            (interaction_1, interaction_2),
            jnp.array([0.0, 1.0]),
        )
        assert multislice.slice_distances.shape[0] == 2

        illumination_field = CoherentField(
            jnp.zeros((100, 100, 1), dtype=jnp.complex64),
            jnp.array(1.0),
            SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0)),
        )
        output_field = resolve_parametrizations(multislice, index=0)(illumination_field)
        assert output_field().shape == (100, 100, 1)
        assert output_field().shape == (100, 100, 1)

    def test_basic_init_call(self) -> None:
        from ptyrax.models.interaction import MultiSlice

        sampling = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0))
        coordinates = CoordinateSystem(
            translation=jnp.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]),
            rotation=Rotation.from_matrix(jnp.tile(jnp.eye(3), (2, 1, 1))),
        )
        multislice = MultiSlice(coordinates, sampling, jnp.array([0.0, 1.0]))
        illumination_field = CoherentField(
            jnp.zeros((100, 100, 1), dtype=jnp.complex64),
            jnp.array(1.0),
            SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0)),
        )
        output_field = resolve_parametrizations(multislice, index=0)(illumination_field)
        assert output_field().shape == (100, 100, 1)

    def test_propagation_dependence(self) -> None:
        from ptyrax.models.interaction import MultiSlice

        sampling = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0))
        coordinates = CoordinateSystem(
            translation=jnp.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]),
            rotation=Rotation.from_matrix(jnp.tile(jnp.eye(3), (2, 1, 1))),
        )
        multislice1 = MultiSlice(coordinates, sampling, jnp.array([0.0, 0.0]))
        multislice2 = MultiSlice(coordinates, sampling, jnp.array([0.0, 10.2]))
        illumination_field = CoherentField(
            jnp.ones((100, 100, 1), dtype=jnp.complex64),
            jnp.array(1.0),
            SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0)),
        )
        output_field1 = resolve_parametrizations(multislice1, index=0)(illumination_field)
        output_field2 = resolve_parametrizations(multislice2, index=0)(illumination_field)
        assert (~jnp.isclose(output_field1(), output_field2())).any()

    def test_slice_distance_gradients(self) -> None:
        from ptyrax.models.interaction import MultiSlice

        sampling = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0))
        coordinates = CoordinateSystem(
            translation=jnp.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]),
            rotation=Rotation.from_matrix(jnp.tile(jnp.eye(3), (2, 1, 1))),
        )
        multislice1 = MultiSlice(coordinates, sampling, jnp.array([0.0, 0.85]))
        illumination_field = CoherentField(
            jnp.ones((100, 100, 1), dtype=jnp.complex64),
            jnp.array(1.0),
            SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0)),
        )

        def loss(multislice: MultiSlice) -> jnp.ndarray:
            output_field = resolve_parametrizations(multislice, index=0)(illumination_field)
            return jnp.sum(jnp.abs(output_field()) ** 2)

        grads = eqx.filter_grad(loss)(multislice1)
        assert np.isfinite(jnp.linalg.norm(grads.slice_distances))
        assert (jnp.linalg.norm(grads.slice_distances) > 0.0).all()

    def test_quarter_lambda(self) -> None:
        from ptyrax.initializers import uniform
        from ptyrax.models.interaction import MultiSlice

        sampling = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0))
        coordinates = CoordinateSystem(
            translation=jnp.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]),
            rotation=Rotation.from_matrix(jnp.tile(jnp.eye(3), (2, 1, 1))),
        )
        multislice1 = MultiSlice(coordinates, sampling, jnp.array([0.0, 0.25]), initializer=uniform)
        illumination_field = CoherentField(
            jnp.ones((50, 50, 1), dtype=jnp.complex64),
            jnp.array(1.0),
            SamplingGrid.from_tuples(shape=(50, 50), pixel_size=(1.0, 1.0)),
        )
        output_field = resolve_parametrizations(multislice1, index=0)(illumination_field)
        np.testing.assert_allclose(output_field()[25, 25, 0], 0.0, atol=1e-5)

    def test_quarter_lambda_tilted_field(self) -> None:
        from ptyrax.initializers import uniform
        from ptyrax.models.interaction import MultiSlice

        tilt_angle = jnp.deg2rad(60.0)
        sampling = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0))
        coordinates = CoordinateSystem(
            translation=jnp.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]),
            rotation=Rotation.from_matrix(jnp.tile(jnp.eye(3), (2, 1, 1))),
        )
        multislice = MultiSlice(coordinates, sampling, jnp.array([0.0, 0.5]), initializer=uniform)
        illumination_field = CoherentField(
            jnp.ones((50, 50, 1), dtype=jnp.complex64),
            jnp.array(1.0),
            SamplingGrid.from_tuples(shape=(50, 50), pixel_size=(1.0, 1.0)),
            propagation_direction=jnp.array([jnp.sin(tilt_angle), 0.0, jnp.cos(tilt_angle)]),
        )
        output_field = resolve_parametrizations(multislice, index=0)(illumination_field)
        np.testing.assert_allclose(output_field()[25, 25, 0], 0.0, atol=1e-5)

    def test_quarter_lambda_tilted_interaction_and_field(self) -> None:
        from ptyrax.initializers import uniform
        from ptyrax.models.interaction import MultiSlice

        tilt_angle = 60.0
        sampling = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0))
        interaction_coordinates = CoordinateSystem(
            translation=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            rotation=Rotation.from_matrix(jnp.tile(R_y(tilt_angle), (2, 1, 1))),
        )
        multislice = MultiSlice(interaction_coordinates, sampling, jnp.array([0.0, 0.5]), initializer=uniform)
        field_coordinates = CoordinateSystem(
            translation=jnp.array([0.0, 0.0, 0.0]),
            rotation=Rotation.from_matrix(R_y(tilt_angle)),
        )

        illumination_field = CoherentField(
            jnp.ones((50, 50, 1), dtype=jnp.complex64),
            jnp.array(1.0),
            SamplingGrid.from_tuples(shape=(50, 50), pixel_size=(1.0, 1.0)),
            coordinate_system=field_coordinates,
        )
        output_field = resolve_parametrizations(multislice, index=0)(illumination_field)
        samples = output_field()

        np.testing.assert_allclose(samples[25, 25, 0], 0.0, atol=1e-5)


def test_to_single_slice_approximation_quarter_lambda() -> None:
    from ptyrax.initializers import uniform
    from ptyrax.models.interaction import MultiSlice, to_single_slice_approximation

    sampling = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0))
    coordinates = CoordinateSystem(
        translation=jnp.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]),
        rotation=Rotation.from_matrix(jnp.tile(jnp.eye(3), (2, 1, 1))),
    )
    multislice1 = MultiSlice(coordinates, sampling, jnp.array([0.0, 0.25]), initializer=uniform)
    approximation = to_single_slice_approximation(multislice1, tilt_angle=0.0, wavelength=1.0)
    assert jnp.allclose(approximation.reflection_coefficient, 0.0)


def test_to_single_slice_approximation_quarter_lambda_tilted() -> None:
    from ptyrax.initializers import uniform
    from ptyrax.models.interaction import MultiSlice, to_single_slice_approximation

    sampling = SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0))
    coordinates = CoordinateSystem(
        translation=jnp.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]),
        rotation=Rotation.from_matrix(jnp.tile(jnp.eye(3), (2, 1, 1))),
    )
    multislice1 = MultiSlice(coordinates, sampling, jnp.array([0.0, 0.7309511]), initializer=uniform)
    approximation = to_single_slice_approximation(multislice1, tilt_angle=70.0, wavelength=1.0)
    assert jnp.allclose(approximation.reflection_coefficient, 0.0)


class TestFresnelReflection2D:
    def test_zero_illumination(self) -> None:
        from ptyrax.models.interaction import FresnelReflection

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
        from ptyrax.models.interaction import FresnelReflection

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
        from ptyrax.models.interaction import FresnelReflection

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
        from ptyrax.models.interaction import FresnelReflection

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
        from ptyrax.models.interaction import FresnelReflection

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
