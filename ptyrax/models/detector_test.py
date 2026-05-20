import jax.numpy as jnp

from ptyrax.spatial import CoordinateSystem, Rotation, SamplingGrid


def test_import_models_detector() -> None:
    __import__("ptyrax.models.detector")


class TestNoiselessDetector:
    def test_init(self) -> None:
        from ptyrax.models.detector import NoiselessEqualWeightDetector

        detector = NoiselessEqualWeightDetector(
            CoordinateSystem(translation=jnp.array([0.0, 0.0, 0.0]), rotation=Rotation()),
            SamplingGrid.from_tuples(shape=(100, 100), pixel_size=(1.0, 1.0)),
        )
        assert jnp.all(detector.sampling.shape == (100, 100))
        assert detector.coordinates.translation.shape == (3,)
        assert detector.coordinates.rotation._representation_6d.shape == (6,)
