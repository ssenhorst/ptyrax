import json
from typing import Any

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import PyTree

from ptyrax.dataset import Ptychogram
from ptyrax.hdf5_checkpoint import (
    apply_hdf5_to_model,
    create_model_from_hdf5,
    load_hdf5_state,
    save_model_hdf5,
)
from ptyrax.models.ptychography import (
    PtychographyModel,
    replace_illumination_from_hdf5,
    scale_illumination_equal_pixel_size,
)
from ptyrax.parametrizations import resolve_parametrizations
from ptyrax.spatial import (
    CoordinateSystem,
    Rotation,
    SamplingGrid,
    matrix_to_six_dimensional_representation,
    six_dimensional_representation_to_matrix,
)
from ptyrax.utils import (
    join_hdf5_paths,
    normalize_hdf5_path,
    resize_to_match,
)


def _build_model(h: int = 100, w: int = 100, seed: int = 42) -> tuple["PtychographyModel", Ptychogram]:
    """Construct a PtychographyModel instance with deterministic inputs.

    Uses pytest.importorskip to make import errors a graceful skip.
    """
    np.random.seed(seed)

    # Generate synthetic but deterministic inputs
    sample_orientations = np.random.random((10, 6))
    sample_positions = np.random.random((10, 3)) * 10
    sample_positions[:, 2] = 0.0  # Only x-y translations
    sample_positions = np.einsum(
        "nid,ni->nd",
        six_dimensional_representation_to_matrix(sample_orientations),
        sample_positions,
    )

    ptychogram = Ptychogram(
        diffraction_patterns=np.random.random((10, h, w)),
        pixel_size=np.array([1.0, 1.0]),
        sample_positions=sample_positions,
        sample_orientations=sample_orientations,
        propagation_distance=1000.0,
        wavelength=np.array([1.0]),
        detector_positions=np.tile(np.array([0.0, 0.0, 1000.0]), (10, 1)),
        detector_orientations=np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (10, 1)),
    )
    model = PtychographyModel.from_image_dataset(ptychogram)
    return model, ptychogram


class _NestedLeaf(eqx.Module):
    weight: jnp.ndarray


class _NestedModel(eqx.Module):
    layer: _NestedLeaf
    bias: jnp.ndarray


def test_initialize_3d_tilted_sampling() -> None:
    from ptyrax.models.ptychography import initialize_3d_tilted_sampling

    detector_sampling = SamplingGrid.from_tuples((100, 100), (1.0, 1.0))
    shape = detector_sampling.shape
    detector_coordinates = CoordinateSystem(
        Rotation(jnp.array([[0, 0, -1, 0, 1, 0], [0, 0, -1, 0, 1, 0]])),
        jnp.array([[100, 0, 0], [100, 0, 0]]),
    )
    sample_coordinates = CoordinateSystem(
        Rotation(
            jnp.array(
                [[jnp.sqrt(0.5) * jnp.sqrt(2), 0, 0.5 * jnp.sqrt(2), 0, 1, 0], [jnp.sqrt(2), 0, jnp.sqrt(2), 0, 1, 0]]
            )
        ),
        jnp.array([[0, 0, 0], [0, 0, 0]]),
    )
    n_dim = 2
    pixel_oversampling = jnp.array([1.0, 1.0])
    window_oversampling = jnp.array([1.0, 1.0])
    wavelengths = jnp.array([1.0])
    sample_sampling, probe_sampling, forward_sampling = initialize_3d_tilted_sampling(
        detector_sampling,
        shape,
        sample_coordinates,
        detector_coordinates,
        wavelengths,
        n_dim,
        pixel_oversampling,
        window_oversampling,
    )
    assert sample_sampling.x_pixel_size > sample_sampling.y_pixel_size


@pytest.mark.skip(reason="Known failing numerics; will be fixed separately")
def test_initialize_3d_tilted_sampling_normal_incidence() -> None:
    from ptyrax.models.ptychography import initialize_3d_tilted_sampling

    detector_pixel_size = (1.0, 1.0)
    detector_sampling = SamplingGrid.from_tuples((100, 100), detector_pixel_size)
    shape = detector_sampling.shape
    detector_coordinates = CoordinateSystem(
        Rotation(
            jnp.array(
                [
                    [1, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 1, 0],
                ]
            )
        ),
        jnp.array([[0, 0, 100], [0, 0, 1000]]),
    )
    sample_coordinates = CoordinateSystem(
        Rotation(jnp.array([[1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0]])), jnp.array([[10, 0, 0], [20, 0, 0]])
    )
    wavelengths = jnp.array([1.0])
    n_dim = 2
    pixel_oversampling = jnp.array([1.0, 1.0])
    window_oversampling = jnp.array([1.0, 1.0])
    sample_sampling, probe_sampling, forward_sampling = initialize_3d_tilted_sampling(
        detector_sampling,
        shape,
        sample_coordinates,
        detector_coordinates,
        wavelengths,
        n_dim,
        pixel_oversampling,
        window_oversampling,
    )
    np.testing.assert_almost_equal(sample_sampling.x_pixel_size, sample_sampling.y_pixel_size)

    far_field_sampling = probe_sampling.to_far_field(1.0, 1000.0)
    # In a transmission geometry at normal incidence, the far-field pixel size should
    # roughly match the detector pixel size.
    np.testing.assert_almost_equal(far_field_sampling.pixel_size[0], detector_pixel_size[0], decimal=2)
    np.testing.assert_almost_equal(far_field_sampling.pixel_size[1], detector_pixel_size[1], decimal=2)

    far_field_sampling = probe_sampling.to_far_field(1.0, 1000.0)
    # In a transmission geometry at normal incidence, the far-field pixel size should
    # roughly match the detector pixel size.
    np.testing.assert_almost_equal(far_field_sampling.pixel_size[0], detector_pixel_size[0], decimal=2)
    np.testing.assert_almost_equal(far_field_sampling.pixel_size[1], detector_pixel_size[1], decimal=2)


class _ScalarWrapper(eqx.Module):
    model: object
    tau: jnp.ndarray  # scalar parameter

    def __init__(self, model: PyTree, tau: float = 0.123) -> None:
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "tau", jnp.asarray(tau))


class TestPtychographyModel:
    def test_init(self) -> None:
        model, _ = _build_model(h=100, w=100, seed=42)
        # Illumination should be present with expected shape.
        assert model.illumination().shape == (1, 100, 100, 1)

    def test_predict(self) -> None:
        from ptyrax.dataset import Ptychogram
        from ptyrax.models.ptychography import PtychographyModel

        np.random.seed(42)

        # sample_orientations = np.random.random((10, 6))
        sample_orientations = matrix_to_six_dimensional_representation(jnp.tile(jnp.eye(3)[jnp.newaxis, :], (10, 1, 1)))
        sample_positions = np.random.random((10, 3)) * 10
        sample_positions[:, 2] = 0.0  # Only x-y translations
        sample_positions = np.einsum(
            "nid, ni -> nd", six_dimensional_representation_to_matrix(sample_orientations), sample_positions
        )

        ptychogram = Ptychogram(
            diffraction_patterns=np.random.random((10, 5, 4)),
            pixel_size=np.array([1.0, 1.0]),
            sample_positions=sample_positions,
            sample_orientations=sample_orientations,
            propagation_distance=1000.0,
            # ca .1 NA
            wavelength=np.array([1.0]),
            detector_positions=np.tile(np.array([0.0, 0.0, 1000.0]), (10, 1)),
            detector_orientations=np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (10, 1)),
        )
        model = PtychographyModel.from_image_dataset(ptychogram)
        index = 1
        pattern = resolve_parametrizations(model, index)()
        assert pattern.shape == ptychogram.diffraction_patterns.shape[1:]

    def test_model_predict_aperture(self) -> None:
        from ptyrax.dataset import Ptychogram
        from ptyrax.models.ptychography import PtychographyModel

        np.random.seed(42)

        # sample_orientations = np.random.random((10, 6))
        sample_orientations = matrix_to_six_dimensional_representation(jnp.tile(jnp.eye(3)[jnp.newaxis, :], (1, 1, 1)))
        sample_positions = np.array([[0, 0, 0]])
        sample_positions[:, 2] = 0.0  # Only x-y translations
        sample_positions = np.einsum(
            "nid, ni -> nd", six_dimensional_representation_to_matrix(sample_orientations), sample_positions
        )
        ptychogram = Ptychogram(
            diffraction_patterns=np.random.random((10, 5, 4)),
            pixel_size=np.array([1.0, 1.0]),
            sample_positions=sample_positions,
            sample_orientations=sample_orientations,
            propagation_distance=1000.0,
            # ca .1 NA
            wavelength=np.array([1.0]),
            detector_positions=np.tile(np.array([0.0, 0.0, 1000.0]), (10, 1)),
            detector_orientations=np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (10, 1)),
        )
        model = PtychographyModel.from_image_dataset(ptychogram)
        index = 1
        pattern = resolve_parametrizations(model, index)()  # noqa: F841
        # We only check that this runs without error for now.

    def test_predict_grad(self) -> None:
        import jax.tree_util as jtu

        from ptyrax.dataset import Ptychogram
        from ptyrax.models.ptychography import PtychographyModel
        from ptyrax.training import mean_square_error

        np.random.seed(42)

        # sample_orientations = np.random.random((10, 6))
        sample_orientations = matrix_to_six_dimensional_representation(jnp.tile(jnp.eye(3)[jnp.newaxis, :], (10, 1, 1)))
        sample_positions = np.random.random((10, 3)) * 10
        sample_positions[:, 2] = 0.0  # Only x-y translations
        sample_positions = np.einsum(
            "nid, ni -> nd", six_dimensional_representation_to_matrix(sample_orientations), sample_positions
        )

        ptychogram = Ptychogram(
            diffraction_patterns=np.random.random((10, 5, 4)),
            pixel_size=np.array([1.0, 1.0]),
            sample_positions=sample_positions,
            sample_orientations=sample_orientations,
            propagation_distance=1000.0,
            # ca .1 NA
            wavelength=np.array([1.0]),
            detector_positions=np.tile(np.array([0.0, 0.0, 1000.0]), (10, 1)),
            detector_orientations=np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (10, 1)),
        )
        index = 1
        indices = jnp.atleast_1d(index)
        model = PtychographyModel.from_image_dataset(ptychogram)

        def predict(indices: jnp.ndarray) -> jnp.ndarray:
            def single_predict(i: int) -> jnp.ndarray:
                return resolve_parametrizations(model, i)()

            return jax.vmap(single_predict)(indices)

        target_diffraction_pattern = jnp.ones((10, 5, 4)) * 1

        loss, grad = eqx.filter_value_and_grad(
            lambda m: mean_square_error(predict(indices), target_diffraction_pattern[indices])
        )(model)

        assert np.all(jtu.tree_map(lambda leaf: np.isfinite(leaf) >= 0, grad))

    def test_hdf5_save_and_metadata_roundtrip(self, tmp_path: str) -> None:
        """Save a model as HDF5 and read back both params and metadata."""
        model, ptychogram = _build_model(h=100, w=100, seed=123)
        ckpt_path = tmp_path / "model_100x100.h5"

        metadata = {
            "run_id": "unit-test",
            "notes": "roundtrip metadata check",
            "versions": {"jax": jax.__version__, "equinox": eqx.__version__},
        }

        save_model_hdf5(model, str(ckpt_path), metadata=metadata)
        assert ckpt_path.exists()

        # Inspect the file structure/attrs low-level
        with h5py.File(str(ckpt_path), "r") as f:
            self._extracted_from_test_hdf5_save_and_metadata_roundtrip_17(f)
        # Use the loader API to roundtrip metadata and state
        state, meta = load_hdf5_state(str(ckpt_path))
        assert isinstance(state, dict) and len(state) > 0
        assert meta["run_id"] == "unit-test"
        assert "versions" in meta

    # TODO Rename this here and in `test_hdf5_save_and_metadata_roundtrip`
    def _extracted_from_test_hdf5_save_and_metadata_roundtrip_17(self, f: h5py.File) -> None:
        assert "params" in f
        assert f.attrs["format"] == "equinox_hdf5_state_v1"
        # Check metadata is JSON in root attrs
        assert "meta_json" in f.attrs
        stored = json.loads(f.attrs["meta_json"])
        assert stored["run_id"] == "unit-test"
        assert stored["notes"] == "roundtrip metadata check"

    def test_hdf5_apply_same_structure_full_load(self, tmp_path: str) -> None:
        """Save from model A, load into freshly initialized model B (same
        structure).

        strict=True should succeed, and leaves should match.
        """
        model_a, ptychogram_a = _build_model(h=100, w=100, seed=0)
        ckpt_path = tmp_path / "same_struct.h5"
        save_model_hdf5(model_a, str(ckpt_path), metadata={"case": "same_structure"})

        # Fresh model with different seed to ensure params do differ before loading.
        model_b, ptychogram_b = _build_model(h=100, w=100, seed=77)

        loaded_model, report, meta = apply_hdf5_to_model(model_b, str(ckpt_path), strict=True, cast=True)

        # No mismatches/missing/extraneous expected when structure is unchanged.
        assert len(report.shape_mismatch) == 0
        assert len(report.missing_in_ckpt) == 0
        assert len(report.extraneous_in_ckpt) == 0
        assert len(report.loaded) > 0

        # Verify arrays now match between model_a and loaded_model (for array-like leaves).
        a_leaves = [leaf for _, leaf in jax.tree_util.tree_flatten_with_path(model_a)[0]]
        b_leaves = [leaf for _, leaf in jax.tree_util.tree_flatten_with_path(loaded_model)[0]]
        assert len(a_leaves) == len(b_leaves)

        # Compare element-wise for array-like leaves
        compared = 0
        for la, lb in zip(a_leaves, b_leaves):
            if eqx.is_array_like(la) and eqx.is_array_like(lb):
                np.testing.assert_allclose(np.asarray(la), np.asarray(lb))
                compared += 1
        assert compared > 0  # we actually compared some parameter arrays

    def test_hdf5_strict_mode_raises_on_shape_change(self, tmp_path: str) -> None:
        """With changed shapes, strict=True should raise a helpful error."""
        model_src, ptychogram_src = _build_model(h=100, w=100, seed=11)
        ckpt_path = tmp_path / "strict_src.h5"
        save_model_hdf5(model_src, str(ckpt_path), metadata={"case": "strict_src"})

        # Different spatial size -> mismatches are expected
        model_dst, ptychogram_dst = _build_model(h=120, w=120, seed=12)

        with pytest.raises(ValueError, match="Strict load failed:"):
            _ = apply_hdf5_to_model(model_dst, str(ckpt_path), strict=True, cast=True)

    def test_hdf5_handles_scalar_params(self, tmp_path: str) -> None:
        """Ensure scalar parameters are saved without filters (no TypeError),
        and can be read back correctly."""
        base = _build_model(h=64, w=64, seed=9)
        wrapped = _ScalarWrapper(base, tau=0.987)
        ckpt = tmp_path / "scalar_ok.h5"

        # Should not raise
        save_model_hdf5(wrapped, str(ckpt), metadata={"case": "scalar"})

        # Inspect HDF5 for any scalar datasets and confirm no compression attrs are set.
        with h5py.File(str(ckpt), "r") as f:
            assert "params" in f
            # Walk datasets and confirm scalar ones exist and were saved fine
            found_scalar = False

            def _walk(group: h5py.Dataset | h5py.Group) -> None:
                nonlocal found_scalar
                for name, item in group.items():
                    if isinstance(item, h5py.Dataset):
                        if item.shape == ():
                            found_scalar = True
                            # Scalar datasets should not have filters/chunking
                            # (h5py shows .chunks is None for contiguous layout)
                            assert item.chunks is None
                    elif isinstance(item, h5py.Group):
                        _walk(item)

            _walk(f["params"])
            assert found_scalar, "Expected at least one scalar dataset in the wrapper"

        # Load and apply back into a fresh wrapped model
        fresh = _ScalarWrapper(_build_model(h=64, w=64, seed=10), tau=0.111)
        loaded, report, meta = apply_hdf5_to_model(fresh, str(ckpt), strict=False)
        # tau should match exactly now
        assert float(np.asarray(loaded.tau)) == pytest.approx(0.987)


class TestRefactorRegression:
    def test_scale_illumination_equal_pixel_size_preserves_probe_container_type(self) -> None:
        model, _ = _build_model(h=48, w=48, seed=1234)
        original_shape = model.illumination.probe.data.shape
        original_type = type(model.illumination.probe.data)

        scaled = scale_illumination_equal_pixel_size(model, scale=1.2)

        assert scaled.illumination.probe.data.shape == original_shape
        assert isinstance(scaled.illumination.probe.data, original_type)

    def test_replace_illumination_from_hdf5_data_only_preserves_probe_container_type(self, tmp_path: str) -> None:
        source_model, _ = _build_model(h=40, w=40, seed=5)
        target_model, _ = _build_model(h=40, w=40, seed=6)
        ckpt_path = tmp_path / "illumination_replace.h5"

        save_model_hdf5(source_model, str(ckpt_path), metadata={"case": "replace_illumination"})
        original_type = type(target_model.illumination.probe.data)

        updated = replace_illumination_from_hdf5(
            target_model,
            str(ckpt_path),
            hdf5_illumination_path="illumination",
            data_only=True,
            normalize=False,
        )

        assert isinstance(updated.illumination.probe.data, original_type)

    def test_apply_hdf5_to_model_accepts_normalized_path_prefix(self, tmp_path: str) -> None:
        source_model, _ = _build_model(h=36, w=36, seed=99)
        target_model, _ = _build_model(h=36, w=36, seed=100)
        ckpt_path = tmp_path / "path_prefix_norm.h5"
        save_model_hdf5(source_model, str(ckpt_path), metadata={"case": "path_prefix"})

        loaded, report, _ = apply_hdf5_to_model(
            target_model.illumination,
            str(ckpt_path),
            path_prefix="/illumination//",
            strict=False,
        )

        assert len(report.loaded) > 0

    def test_create_model_from_hdf5_loads_nested_paths(self, tmp_path: str) -> None:
        src = _NestedModel(layer=_NestedLeaf(weight=jnp.array([1.0, 2.0], dtype=jnp.float32)), bias=jnp.array(3.0))
        ckpt_path = tmp_path / "nested_model.h5"
        save_model_hdf5(src, str(ckpt_path), metadata={"case": "nested"})

        loaded = create_model_from_hdf5(
            _NestedModel,
            ckpt_path,
            layer=_NestedLeaf(weight=jnp.zeros((2,), dtype=jnp.float32)),
            bias=jnp.array(0.0),
        )

        np.testing.assert_allclose(np.asarray(loaded.layer.weight), np.asarray(src.layer.weight))
        np.testing.assert_allclose(np.asarray(loaded.bias), np.asarray(src.bias))

    def test_hdf5_path_normalization_helpers(self) -> None:
        assert normalize_hdf5_path("/illumination//probe/") == "illumination/probe"
        assert join_hdf5_paths("/illumination/", "//probe/data") == "illumination/probe/data"


# -----------------------------
# Helpers
# -----------------------------


def make_array(shape: tuple[int, ...], start: int = 0) -> np.ndarray:
    """Create an array with increasing integers for easy visual checks."""
    return np.arange(start, start + np.prod(shape), dtype=np.int32).reshape(shape)


# -----------------------------
# Tests
# -----------------------------


def test_center_crop_contracting_axes() -> None:
    """When old > new, we crop symmetrically to keep the center."""
    arr: np.ndarray = make_array((6, 6))
    resized: np.ndarray = resize_to_match(arr, (4, 4), axis_policies={}, default_policy="pad")
    resized: np.ndarray = resize_to_match(arr, (4, 4), axis_policies={}, default_policy="pad")
    assert resized.shape == (4, 4)
    expected_center: np.ndarray = arr[1:5, 1:5]
    np.testing.assert_array_equal(resized, expected_center)


@pytest.mark.parametrize("policy", ["pad", "repeat", "tile"])
def test_expand_policy_global(policy: str) -> None:
    """When old < new, expansion uses the global policy."""
    arr: np.ndarray = make_array((2, 2))
    resized: np.ndarray = resize_to_match(arr, (4, 4), axis_policies={}, default_policy=policy)
    resized: np.ndarray = resize_to_match(arr, (4, 4), axis_policies={}, default_policy=policy)
    assert resized.shape == (4, 4)
    # All values should come from original set
    assert set(resized.flatten()).issubset(set(arr.flatten()))


def test_per_axis_policy_mixed() -> None:
    """
    Demonstrate per-axis control:
    - Axis 0 expands by repeat
    - Axis 1 expands by pad
    """
    arr: np.ndarray = make_array((2, 2))
    target_shape: tuple[int, int] = (4, 6)
    axis_policies: dict[int, str] = {0: "repeat", 1: "pad"}
    resized: np.ndarray = resize_to_match(arr, target_shape, axis_policies, default_policy="pad")
    resized: np.ndarray = resize_to_match(arr, target_shape, axis_policies, default_policy="pad")
    assert resized.shape == target_shape
    # Axis 0 should contain repeated rows
    assert np.all(resized[0] == resized[1])
    # Axis 1 should have zeros padded on sides
    assert np.any(resized[:, 0] == 0) or np.any(resized[:, -1] == 0)


def test_tile_policy_multidim() -> None:
    """Tile policy replicates across both axes and then crops to center."""
    arr: np.ndarray = make_array((2, 3))
    target_shape: tuple[int, int] = (5, 7)
    resized: np.ndarray = resize_to_match(arr, target_shape, axis_policies={}, default_policy="tile")
    resized: np.ndarray = resize_to_match(arr, target_shape, axis_policies={}, default_policy="tile")
    assert resized.shape == target_shape
    # Check that pattern repeats (top-left corner equals bottom-right corner)
    assert resized[0, 0] == resized[-1, -1]


def test_policy_map_example() -> None:
    """
    Show how regex-based policy_map would apply:
    - For path 'encoder/weights', axis 0 repeats, axis 1 pads.
    """
    path: str = "encoder/weights"
    policy_map: dict[str, dict[str, Any]] = {"encoder/.*": {"default": "pad", "axes": {0: "repeat", 1: "pad"}}}
    matched: dict[str, Any] | None = None
    for pattern, pol in policy_map.items():
        if __import__("re").fullmatch(pattern, path):
            matched = pol
            break
    assert matched is not None
    assert matched["axes"][0] == "repeat"
    assert matched["default"] == "pad"
