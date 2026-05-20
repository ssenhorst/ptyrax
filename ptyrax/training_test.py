import jax.numpy as jnp
import numpy as np
import pytest


def test_mean_square_error() -> None:
    from ptyrax.training import mean_square_error

    a = jnp.array(np.random.uniform(size=(5, 20, 20)))
    b = jnp.array(np.random.uniform(size=(5, 20, 20)))
    c = mean_square_error(a, b)
    assert c < jnp.mean(a**2) + jnp.mean(b**2)


def test_mean_square_error_gradient() -> None:
    from jax import value_and_grad

    from ptyrax.training import mean_square_error

    a = jnp.array(np.random.uniform(size=(5, 20, 20)))
    b = jnp.array(np.random.uniform(size=(5, 20, 20)))
    c, dc = value_and_grad(mean_square_error)(a, b)
    assert np.all(np.isfinite(dc))


def test_set_model_constant_tilt_angle_keeps_detector_geometry() -> None:
    from ptyrax.dataset import Ptychogram
    from ptyrax.models.ptychography import PtychographyModel, set_model_constant_tilt_angle
    from ptyrax.parametrizations import resolve_parametrizations
    from ptyrax.spatial import R_y, matrix_to_six_dimensional_representation, six_dimensional_representation_to_matrix

    n = 4
    ptychogram = Ptychogram(
        diffraction_patterns=np.ones((n, 8, 8), dtype=np.float32),
        pixel_size=np.array([1.0, 1.0]),
        sample_positions=np.zeros((n, 3), dtype=np.float32),
        sample_orientations=np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (n, 1)),
        propagation_distance=np.ones(n, dtype=np.float32) * 2.0,
        wavelength=np.array([1.0], dtype=np.float32),
        detector_positions=np.tile(np.array([0.0, 0.0, 2.0], dtype=np.float32), (n, 1)),
        detector_orientations=np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (n, 1)),
    )

    model = PtychographyModel.from_image_dataset(ptychogram)
    tilted_model = set_model_constant_tilt_angle(model, tilt_angle=10.0)
    tilted_model = resolve_parametrizations(tilted_model)

    sample_orientations = np.array(tilted_model.interaction.coordinates.rotation._representation_6d)
    detector_orientations = np.array(tilted_model.detector.coordinates.rotation._representation_6d)

    expected_detector_orientations = np.array(
        matrix_to_six_dimensional_representation(
            R_y(180)
            @ six_dimensional_representation_to_matrix(sample_orientations)
            @ six_dimensional_representation_to_matrix(sample_orientations)
        )
    )

    np.testing.assert_allclose(detector_orientations, expected_detector_orientations, rtol=1e-5, atol=1e-6)
    assert not np.allclose(detector_orientations, sample_orientations)


@pytest.mark.skip(reason="Lenspaper data is not currently a part of the repository")
def test_make_constant_tilt_angle() -> None:
    from ptyrax.dataset import from_hdf5, make_constant_tilt_angle
    from ptyrax.models.ptychography import PtychographyModel, set_model_constant_tilt_angle

    ptychogram = from_hdf5("data/lenspaper/lenspaper.hdf5")
    model = PtychographyModel(ptychogram)
    model = set_model_constant_tilt_angle(model, tilt_angle=10.0)

    ptychogram = make_constant_tilt_angle(ptychogram, tilt_angle=np.array([0, 10.0, 0.0]))
    model_2 = PtychographyModel(ptychogram)
    assert np.allclose(model.interaction.surface_normal, model_2.interaction.surface_normal)
    assert np.allclose(model.detector.coordinates.translation, model_2.detector.coordinates.translation)
    assert np.allclose(
        model.detector.coordinates.rotation._representation_6d, model_2.detector.coordinates.rotation._representation_6d
    )
    assert np.allclose(model.interaction.coordinates.translation, model_2.interaction.coordinates.translation)
    assert np.allclose(
        model.interaction.coordinates.rotation._representation_6d,
        model_2.interaction.coordinates.rotation._representation_6d,
    )
