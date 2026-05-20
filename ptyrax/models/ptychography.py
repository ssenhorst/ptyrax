# checkpointing.py
# hdf5_checkpointing.py
from __future__ import annotations

import functools
import logging
import os
import re
import warnings  # for safe HDF5 path components
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Type, Union

import equinox as eqx
import gin
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import vmap
from jax.scipy.spatial.transform import Rotation as JaxRotation
from jaxtyping import Array, Bool, Complex, Float, Inexact, Key, PyTree, ScalarLike, Shaped
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.image import AxesImage
from tensorboardX import SummaryWriter

from ptyrax.dataset import ImageDataset, Ptychogram
from ptyrax.field import (
    CoherentField,
)
from ptyrax.hdf5_checkpoint import (
    apply_hdf5_to_model,
    load_hdf5_state,
)
from ptyrax.initializers import aperture, uniform
from ptyrax.models.detector import (
    BackgroundEqualWeightDetector,
    Detector,
)
from ptyrax.models.illumination import (
    DirectIllumination,
    IlluminationModel,
    NamedLoss,
)
from ptyrax.models.interaction import (
    FresnelReflection,
    InteractionModel,
    MultiSlice,
)
from ptyrax.models.propagation import (
    FarfieldPropagator,
    Propagator,
)
from ptyrax.parametrizations import (
    ArrayParametrization,
    DirectArrayParametrization,
    IndexSliceParameter,
    NormalizedReferencedArrayParametrization,
    resolve_index_dependent_parameters,
    resolve_parametrizations,
)
from ptyrax.spatial import (
    CoordinateSystem,
    R_y,
    Rotation,
    SamplingGrid,
    interpolate_grid_to_grid,
    matrix_to_six_dimensional_representation,
    plot_geometry,
    shift_with_interpolation_unequal_pixel_size,
    six_dimensional_representation_to_matrix,
)
from ptyrax.state_geometry import (
    coherent_field_from_hdf5_state,
    state_get_optional,
    state_get_with_candidates,
    state_pick_index,
)
from ptyrax.utils import (
    compute_center_of_mass_shift,
    fft,
    identity,
    ifft,
    make_length_n,
    make_or_reuse_axes,
    make_path_string,
    phase_only_exp,
    plot,
    set_probe_data_preserve_parametrization,
    shift_image,
)

X_AXIS = jnp.array([1.0, 0.0, 0.0])
Y_AXIS = jnp.array([0.0, 1.0, 0.0])
Z_AXIS = jnp.array([0.0, 0.0, 1.0])

_state_get_with_candidates = state_get_with_candidates
_state_pick_index = state_pick_index
_state_get_optional = state_get_optional


def plot_spheres_xz(
    sample_rotation_matrix: Float[Array, "n 3 3"],
    detector_coordinate_sphere: Float[Array, "n m 3"],
    illumination_coordinate_sphere: Float[Array, "n k 3"],
    scattering_vector: Float[Array, "n m k 3"],
    fig: plt.Figure = None,
    gs: plt.GridSpec = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot detector, illumination, and scattering vectors on the Ewald sphere
    in the XZ plane.

    Visualizes the Fourier-space geometry of the ptychography experiment by projecting
    the detector and illumination direction cosines, as well as the resulting scattering
    vectors, into the sample-frame XZ plane ($\\xi_x$ vs $\\xi_z$).

    Args:
        sample_rotation_matrix: Rotation matrices transforming from lab to sample frame,
            shape ``(n, 3, 3)``.
        detector_coordinate_sphere: Unit vectors towards detector pixels in lab frame,
            shape ``(n, m, 3)``.
        illumination_coordinate_sphere: Unit vectors towards illumination directions in
            lab frame, shape ``(n, k, 3)``.
        scattering_vector: Difference vectors (detector - illumination) representing
            momentum transfer, shape ``(n, m, k, 3)``.
        fig: Existing matplotlib figure to draw into. If ``None``, a new figure is created.
        gs: A :class:`~matplotlib.gridspec.SubplotSpec` for axis placement within ``fig``.

    Returns:
        A tuple of ``(figure, axes)`` containing the plotted geometry.
    """

    def sf2(c: Float[Array, " s n m d"]) -> Float[Array, " s n m i"]:
        return jnp.einsum("sid, snmd -> snmi", sample_rotation_matrix, c)

    def sf1(c: Float[Array, " s n d"]) -> Float[Array, " s n i"]:
        return jnp.einsum("sid, snd -> sni", sample_rotation_matrix, c)

    fig, ax = make_or_reuse_axes(fig, gs)
    ax.plot(
        sf1(detector_coordinate_sphere)[0, :, 0],
        sf1(detector_coordinate_sphere)[0, :, 2],
        ".",
    )
    ax.plot(
        sf1(illumination_coordinate_sphere)[0, :, 0],
        sf1(illumination_coordinate_sphere)[0, :, 2],
        ".",
    )
    ax.plot(
        sf2(scattering_vector)[0, :, :, 0].flatten(),
        sf2(scattering_vector)[0, :, :, 2].flatten(),
        ".",
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("$\\xi_x$")
    ax.set_ylabel("$\\xi_z$")
    return fig, ax


def plot_scattering_vector_xy(
    scattering_vector: Float[Array, "s n m d"],
    sample_rotation_matrix: Float[Array, "s 3 3"],
    fig: plt.Figure = None,
    gs: plt.GridSpec = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot Fourier scattering vectors in the sample-frame XY plane.

    Projects the scattering vectors (momentum transfer) into the sample coordinate
    system and displays their $\\xi_x$ vs $\\xi_y$ components. This is useful for
    verifying lateral Fourier coverage of the experiment.

    Args:
        scattering_vector: Scattering vectors in lab frame, shape ``(s, n, m, d)``.
        sample_rotation_matrix: Rotation matrices from lab to sample frame,
            shape ``(s, 3, 3)``.
        fig: Existing matplotlib figure to draw into. If ``None``, a new figure is created.
        gs: A :class:`~matplotlib.gridspec.SubplotSpec` for axis placement within ``fig``.

    Returns:
        A tuple of ``(figure, axes)`` containing the plotted geometry.
    """

    def sf2(c: Float[Array, " s n m d"]) -> Float[Array, " s n m i"]:
        return jnp.einsum("sid, snmd -> snmi", sample_rotation_matrix, c)

    fig, ax = make_or_reuse_axes(fig, gs)
    ax.plot(
        sf2(scattering_vector)[0, :, :, 0].flatten(),
        sf2(scattering_vector)[0, :, :, 1].flatten(),
        ".",
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("$\\xi_x$")
    ax.set_ylabel("$\\xi_y$")
    return fig, ax


# Detector coordinate system follows convention from CXI data format:
# The illumination is originating from the local -z direction.
# The origin is at the center of the detector. The first coordinate
# corresponds to x, the last to y. Pixel (0,0) is at positive
# x and y, pixel (0,1) is at positive x and negative y, etc.


class ImagePredictionModel(eqx.Module):
    """Abstract base for models that predict images (or equivalently simulates
    detectors in an experiment). Since this is an instance of eqx.Module, any
    instance of an image prediction model is also a pytree. It is therefore
    allowed to pass instances of this class to jitted functions without any
    hassle.

    To ensure functional purity, instances of this class are frozen after initialization. This means that no fields of
    the instance can be modified after initialization. Instead, to modify any fields, a new instance must be created
    with the desired changes. This can be done using the `eqx.tree_at` function. For more information, see the `Equinox
    documentation <https://docs.kidger.site/equinox/api/manipulation/#equinox.tree_at>`_.

    Since this is a dataclass, fields may be defined using annotations. If any of the fields are jax Arrays but should
    not be optimized over, **do not** mark them as static using `eqx.field`, this will lead to errors when trying to
    jit functions using the model. Instead, leave them as normal fields, and make sure that the optimizer specification
    does not include these parameters. Static parameters are allowed for non-array fields, such as static shapes,
    configuration constants, etc.
    """

    @abstractmethod
    def __call__(self) -> Float[Array, " m n"]:
        """The main part to predict images from the model. This will be called
        inside the jit-boundary, so all operations contained herein must be
        jittable. The output must be a single image of shape (m, n). Before
        this function is called, any parts of the model which are types of
        parametrizations will be evaluated to their underlying arrays.
        Therefore, inside of this function, all parametrizations can be treated
        as normal arrays. For example, any model parameters which are instances
        of IndexSliceParametrization (with leading dimension `d` for the
        dataset index), will have their leading dimension removed inside of
        `ImagePredictionModel.__call__()`. This way, the model needs not worry
        about including batch dimensions at all, every prediction should be
        just for a single image.

        Returns:
            Float[Array, " m n"]: The predicted image of shape (m, n).
        """
        pass

    @classmethod
    @abstractmethod
    def from_image_dataset(
        cls,
        dataset: ImageDataset,
        *args,
        **kwargs,
    ) -> "ImagePredictionModel":
        """Instantiates the ImagePredictionModel based on its corresponding
        dataset. This will be called outside the jit-boundary, so all fields of
        the ImageDataset are likely numpy arrays.

        Args:
            dataset (ImageDataset): An instance of the ImageDataset which the model will attempt to predict. When
            implementing ImagePredictionModel, this usually should also mean implementing a corresponding ImageDataset.

        Returns:
            ImagePredictionModel: The model that will be used in the optimization loop to predict images from the
            dataset.
        """
        pass

    @abstractmethod
    def to_image_dataset(self, predicted_images: Shaped[Array, "* m n"]) -> ImageDataset:
        """Converts predicted images back to an ImageDataset. This is mainly
        useful for evaluation and logging purposes.

        Args:
            predicted_images (Float[Array, "* m n"]): The predicted images from the model. These will likely come from
            the output of a simulation.

        Returns:
            ImageDataset: An instance of the ImageDataset containing the predicted images.
        """
        pass

    @property
    @abstractmethod
    def image_shape(self) -> tuple[int, int]:
        """

        Returns:
            tuple[int, int]: The shape of a single image which the model predicts.
        """
        pass

    def resolve(self, *args) -> "ImagePredictionModel":
        """Resolves all parametrizations in the model to their underlying
        arrays.

        Returns:
            ImagePredictionModel: A new instance of the model with all parametrizations resolved to arrays.
        """
        return resolve_parametrizations(self, *args)

    def __log_epoch__(
        self,
        writer: SummaryWriter,
        epoch: int,
        prefix: str = "",
        **kwargs,
    ) -> None:
        """May be overwritten to log relevant model parameters to TensorBoard.
        For functions to log images, see `ptyrax.logger`.

        Args:
            writer (SummaryWriter): The TensorBoard SummaryWriter to log to.
            epoch (int): The current epoch number.
            prefix (str, optional): Prefix to add to all logged parameter names. Defaults to "".
        """
        pass

    def __regularize__(self) -> tuple[float, list[NamedLoss]]:
        """May be overwritten to provide model-specific regularization terms.

        Returns:
            tuple[float, list[NamedLoss]]: A tuple containing the total regularization loss and a list of NamedLoss
            instances for logging individual regularization contributions.
        """
        return 0.0, []

    def print_parameter_paths(self, prefix: str = "") -> None:
        """Prints all parameters of the model to the console.

        This is useful for debugging and for getting an overview of
        the model's parameters. It uses the `ptyrax.utils.print_parameters` function, which recursively prints all
        parameters in a readable format.
        """

        jax.tree.map_with_path(
            lambda path, x: print(f"{prefix}{'.'.join([p.name for p in path])}"),  # noqa: T201
            self,
        )

    def save(self, file_path: Path) -> None:
        """Serialize the model's leaves to disk using Equinox serialization.

        Saves all array leaves of the model (parameters, buffers) to a binary file
        that can later be loaded with :py:meth:`~ptyrax.models.ptychography.ImagePredictionModel.load`.

        Args:
            file_path: Destination file path (typically with ``.eqx`` extension).
        """
        eqx.tree_serialise_leaves(file_path, self)

    def load(self, file_path: Path) -> "PtychographyModel":
        """Deserialize model leaves from disk into the current model structure.

        The current model instance acts as the structural template (pytree skeleton)
        and its leaf values are replaced with the serialized values from ``file_path``.

        Args:
            file_path: Path to a previously saved ``.eqx`` file.

        Returns:
            A new model instance with the same structure as ``self`` but with
            leaf values loaded from the file.
        """
        model_loaded = eqx.tree_deserialise_leaves(file_path, self)
        return model_loaded

    @property
    def n_indices(self) -> int:
        """The number of dataset indices (scan positions) this model spans.

        Determined by inspecting all :py:class:`~ptyrax.parametrizations.IndexSliceParameter`
        leaves in the model tree and returning the maximum leading dimension found.
        If no ``IndexSliceParameter`` fields exist, returns 1 with a warning.

        Returns:
            The number of dataset indices the model is configured for.
        """

        def get_number_of_indices(leaf: Any) -> int:  # noqa: ANN401
            if leaf is None or not isinstance(leaf, IndexSliceParameter):
                return 0
            return leaf.n

        indices = jax.tree.map(get_number_of_indices, self, is_leaf=lambda x: isinstance(x, IndexSliceParameter))
        max_index = jax.tree.reduce_associative(lambda a, b: max(a, b), indices)
        if max_index == 0:
            warnings.warn(
                "The model does not seem to have any IndexSliceParametrization fields. The 'n_indices' property will "
                "return 1. If this is unintended, please ensure that the model uses IndexSliceParametrization for "
                "dataset-indexed parameters.",
                UserWarning,
            )
            return 1
        return max_index


default_policy_map = {
    r".*": {"default": "pad"},
}


@gin.configurable
def load_model_from_reconstruction(
    model: ImagePredictionModel,
    reconstruction_path: str,
    policy_map: dict = default_policy_map,
    **kwargs,
) -> ImagePredictionModel:
    """Load model weights/state from a previous reconstruction HDF5 into a new
    model.

    Args:
        dataset: Dataset used to construct the model shape.
        reconstruction_path: Path to the HDF5 file containing saved model parameters.
        policy_map: Mapping of regex patterns to handling policies when applying HDF5.

    Returns:
        An `ImagePredictionModel` instance with parameters loaded from `reconstruction_path`.
    """
    model, _, _ = apply_hdf5_to_model(model, reconstruction_path, policy_map=policy_map)
    return model


@gin.configurable
def preprocess_model(
    model: ImagePredictionModel,
    preprocess_functions: tuple[Callable[[ImagePredictionModel], ImagePredictionModel], ...] = (),
) -> ImageDataset:
    """Apply a sequence of preprocessing functions to the model.

    Args:
        model: The `ImagePredictionModel` or model to preprocess.
        preprocess_functions: Tuple of callables applied in order. Each callable should take an `ImagePredictionModel`
        and return a processed `ImagePredictionModel`.

    Returns:
        The processed `ImagePredictionModel`
    """
    for adjuster in preprocess_functions:
        if adjuster is None:
            continue
        model = adjuster(model)
    return model


# endregion
# region Ptychography models
# region Main model


@gin.configurable
class PtychographyModel(ImagePredictionModel):
    """Top-level ptychography model composed of `illumination`, `interaction`,
    `propagator`, and `detector`.

    The classmethod `from_image_dataset` initializes model components from a `Ptychogram` dataset.
    """

    illumination: IlluminationModel
    interaction: InteractionModel
    propagator: Propagator
    detector: Detector

    @classmethod
    @gin.configurable("PtychographyModel_initializer")
    def from_image_dataset(
        cls,
        ptychogram: Ptychogram,
        illumination_class: type[IlluminationModel] = DirectIllumination,
        probe_initializer: Callable[[SamplingGrid], Complex[Array, "n d"]] = aperture,
        interaction_class: type[InteractionModel] = FresnelReflection,
        interaction_initializer: Callable[[SamplingGrid], Complex[Array, "n d"]] = uniform,
        detector_class: type[Detector] = BackgroundEqualWeightDetector,
        propagator_class: type[Propagator] = FarfieldPropagator,
        *,
        tensorboard_writer: SummaryWriter = None,
        key: Key = jax.random.PRNGKey(42),
        fixed_sampling: tuple[int, int, int, int] = None,
    ) -> None:
        """Construct a :py:class:`PtychographyModel` from a ptychography
        dataset.

        This factory method computes the experimental geometry (sample/detector
        coordinate systems), determines the Fourier sampling grids from the scattering
        geometry, initializes the illumination probe and interaction model, and
        assembles all components into a complete forward model.

        Args:
            ptychogram: The :py:class:`~ptyrax.dataset.Ptychogram` dataset containing
                diffraction patterns, scan positions, and experimental metadata.
            illumination_class: Class to use for the illumination model.
            probe_initializer: Callable that generates initial probe field data on a
                given :py:class:`~ptyrax.spatial.SamplingGrid`.
            interaction_class: Class to use for the sample interaction model.
            interaction_initializer: Callable that generates initial interaction
                (e.g. reflection coefficient) data.
            detector_class: Class to use for the detector model.
            propagator_class: Class to use for the field propagator.
            tensorboard_writer: Optional TensorBoard writer for logging sampling
                geometry during initialization.
            key: JAX PRNG key for stochastic initialization.
            fixed_sampling: If provided, a tuple ``(interaction_shape, interaction_pixel_size,
                probe_shape, probe_pixel_size)`` that bypasses automatic sampling
                computation.

        Returns:
            A fully initialized :py:class:`PtychographyModel`.
        """
        propagator = propagator_class()
        wavelengths = jnp.asarray(ptychogram.wavelength)
        sample_coordinates = cls.initialize_sample_coordinates(ptychogram)
        detector_coordinates = cls.initialize_detector_coordinates(ptychogram)

        if fixed_sampling is not None:
            interaction_grid = SamplingGrid.from_tuples(fixed_sampling[0], fixed_sampling[1])
            probe_grid = SamplingGrid.from_tuples(fixed_sampling[2], fixed_sampling[3])
        else:
            logging.info("Initializing sampling...")
            detector_grid = SamplingGrid(
                ptychogram.pixel_number,
                ptychogram.pixel_size.reshape((2,)),
            )
            interaction_grid, probe_grid, forward_grid = initialize_3d_tilted_sampling(
                detector_grid,
                detector_grid.shape,
                sample_coordinates,
                detector_coordinates,
                wavelengths,
                writer=tensorboard_writer,
            )

        logging.info(f"Initialized sampling:\nSample: {interaction_grid}\nProbe: {probe_grid}\n")

        @eqx.filter_vmap(in_axes=(0, 0, 0))
        def initialize_single_wavelength_illumination(
            wavelength: float, weight: float, key: Array
        ) -> IlluminationModel:
            probe_data = probe_initializer(
                probe_grid,
                weight=weight,
                pixel_size=probe_grid.pixel_size,
                dtype=jnp.complex64,
                key=key,
            )
            probe_data = probe_data[..., np.newaxis]
            probe_coordinates = CoordinateSystem(
                rotation=resolve_parametrizations(sample_coordinates).rotation[0],
                translation=jnp.array([0.0, 0.0, 0.0]),
            )

            return illumination_class(
                probe_data,
                wavelength,
                probe_grid,
                probe_coordinates,
                n_scan=ptychogram.n,
                key=key,
            )

        illumination_init_keys = jax.random.split(key, wavelengths.shape[0])
        illumination_init_weights = jnp.asarray(np.linspace(0, 1, wavelengths.shape[0]))

        illumination = initialize_single_wavelength_illumination(
            wavelengths, illumination_init_weights, illumination_init_keys
        )

        interaction = interaction_class(
            sample_coordinates, interaction_grid, forward_grid, initializer=interaction_initializer
        )

        detector = detector_class(
            coordinates=detector_coordinates,
            sampling=detector_grid,
            dark_counts=ptychogram.detector_darkframe,
            scale=ptychogram.diffraction_pattern_scale,
        )
        return cls(
            illumination=illumination,
            interaction=interaction,
            propagator=propagator,
            detector=detector,
        )

    @classmethod
    def from_hdf5_state(
        cls,
        state: Dict[str, np.ndarray],
        *,
        illumination_class: type[IlluminationModel] = DirectIllumination,
        interaction_class: type[InteractionModel] = FresnelReflection,
        detector_class: type[Detector] = BackgroundEqualWeightDetector,
        propagator_class: type[Propagator] = FarfieldPropagator,
    ) -> "PtychographyModel":
        """Reconstruct a :py:class:`PtychographyModel` from a flat HDF5 state
        dictionary.

        This method rebuilds the model's component hierarchy (illumination,
        interaction, detector, propagator) from a dictionary of named arrays
        typically produced by :py:func:`~ptyrax.hdf5_checkpoint.load_hdf5_state`.

        Args:
            state: Dictionary mapping HDF5 dataset paths to numpy arrays, as returned
                by :func:`~ptyrax.hdf5_checkpoint.load_hdf5_state`.
            illumination_class: Class to use for the illumination model. Must implement
                a ``from_coherent_field`` classmethod.
            interaction_class: Class to use for the sample interaction model.
            detector_class: Class to use for the detector model. Must implement
                a ``from_hdf5_state`` classmethod.
            propagator_class: Class to use for the field propagator.

        Returns:
            A :py:class:`PtychographyModel` with parameters populated from the HDF5 state.

        Raises:
            TypeError: If ``illumination_class`` does not implement ``from_coherent_field``.
            ValueError: If required keys are missing from ``state``.
        """

        probe = coherent_field_from_hdf5_state(state, index=0, probe_path_prefix="illumination/_probe")

        if not hasattr(illumination_class, "from_coherent_field"):
            raise TypeError(
                f"illumination_class={illumination_class.__name__} must implement from_coherent_field for HDF5 loading."
            )
        illumination = illumination_class.from_coherent_field(probe)

        interaction_rotation = np.asarray(
            _state_get_with_candidates(
                state,
                [
                    "interaction/coordinates/parameters/rotation/_representation_6d",
                    "interaction/coordinates/rotation/_representation_6d",
                ],
            )
        )
        interaction_translation = np.asarray(
            _state_get_with_candidates(
                state,
                [
                    "interaction/coordinates/parameters/_translation",
                    "interaction/coordinates/_translation",
                    "interaction/coordinates/translation",
                ],
            )
        )
        interaction_coordinates = IndexSliceParameter(
            CoordinateSystem(
                rotation=Rotation(jnp.asarray(interaction_rotation)),
                translation=jnp.asarray(interaction_translation),
                model_in_local_frame=True,
            )
        )

        interaction_sampling_pixel_size = np.asarray(
            _state_get_with_candidates(
                state,
                ["interaction/sampling/pixel_size"],
            )
        ).reshape(-1)
        if interaction_sampling_pixel_size.size == 1:
            interaction_sampling_pixel_size = np.repeat(interaction_sampling_pixel_size, 2)

        interaction_forward_pixel_size = np.asarray(
            _state_get_with_candidates(
                state,
                ["interaction/forward_sampling/pixel_size"],
            )
        ).reshape(-1)
        if interaction_forward_pixel_size.size == 1:
            interaction_forward_pixel_size = np.repeat(interaction_forward_pixel_size, 2)

        reflection_coefficient = np.asarray(
            _state_get_with_candidates(
                state,
                [
                    "interaction/reflection_coefficient/_data",
                    "interaction/inner_interactions/reflection_coefficient/_data",
                ],
            )
        )
        interaction_shape = tuple(int(v) for v in reflection_coefficient.shape[-2:])
        interaction_sampling = SamplingGrid.from_tuples(
            interaction_shape,
            tuple(float(v) for v in interaction_sampling_pixel_size[:2]),
        )
        interaction_forward_shape_arr = _state_get_optional(
            state,
            [
                "interaction/forward_sampling/shape",
                "interaction/forward_sampling/_shape",
            ],
        )
        if interaction_forward_shape_arr is not None:
            interaction_forward_shape_arr = np.asarray(interaction_forward_shape_arr).reshape(-1)
            if interaction_forward_shape_arr.size == 1:
                interaction_forward_shape_arr = np.repeat(interaction_forward_shape_arr, 2)
            if interaction_forward_shape_arr.size < 2:
                raise ValueError(
                    "Forward sampling shape under 'interaction/forward_sampling/shape' must provide at least one value."
                )
            interaction_forward_shape = tuple(int(v) for v in interaction_forward_shape_arr[:2])
        else:
            interaction_forward_shape = tuple(int(v) for v in probe.sampling.shape[:2])

        interaction_forward_sampling = SamplingGrid.from_tuples(
            interaction_forward_shape,
            tuple(float(v) for v in interaction_forward_pixel_size[:2]),
        )

        interaction = interaction_class(
            coordinates=interaction_coordinates,
            sampling=interaction_sampling,
            forward_sampling=interaction_forward_sampling,
            initializer=lambda _sampling: jnp.asarray(reflection_coefficient),
        )

        detector = detector_class.from_hdf5_state(state)
        propagator = propagator_class()

        return cls(
            illumination=illumination,
            interaction=interaction,
            propagator=propagator,
            detector=detector,
        )

    @classmethod
    def from_hdf5(
        cls,
        file_path: str | os.PathLike | h5py.File | h5py.Group,
        *,
        params_root: str = "params",
        **kwargs,
    ) -> "PtychographyModel":
        """Instantiate a ptychography model from an HDF5 file/group params
        subtree."""
        state, _ = load_hdf5_state(file_path, params_root=params_root)
        return cls.from_hdf5_state(state, **kwargs)

    def exit_field(self, index: int = 0) -> CoherentField:
        """Return the field immediately after the interaction model for a given
        dataset index."""
        resolved_model = resolve_index_dependent_parameters(self, index=index)
        resolved_model = resolve_parametrizations(resolved_model)

        probe = resolved_model.illumination()
        if probe.data.ndim > 3 and probe.data.shape[0] > 1:
            probe = probe[0]
        return resolved_model.interaction(probe)

    def to_image_dataset(self, predicted_diffraction_patterns: Float[Array, "* m n"]) -> Ptychogram:
        """Convert model state and predicted diffraction patterns into a
        :py:class:`~ptyrax.dataset.Ptychogram`.

        Packs the predicted images together with the model's current geometric
        parameters (sample/detector positions, orientations, wavelength, pixel size)
        into a dataset suitable for saving or comparison with measured data.

        Args:
            predicted_diffraction_patterns: Predicted intensity patterns with shape
                matching the number of scan positions and detector pixels.

        Returns:
            A :py:class:`~ptyrax.dataset.Ptychogram` populated with predictions and
            the model's current geometry.
        """
        interaction_coordinates = resolve_parametrizations(self.interaction.coordinates)
        detector_coordinates = resolve_parametrizations(self.detector.coordinates)
        # propagation_distance is the distance from sample to detector
        # It should equal the norm of detector positions (per-position array)
        propagation_distances = np.linalg.norm(detector_coordinates.translation, axis=-1)

        # Verify that propagation distance is constant across all positions
        propagation_distance_mean = np.mean(propagation_distances)
        if not np.allclose(propagation_distances, propagation_distance_mean, rtol=1e-6):
            logging.warning(
                f"Propagation distance varies across positions! "
                f"min={np.min(propagation_distances):.6e}, max={np.max(propagation_distances):.6e}, "
                f"mean={propagation_distance_mean:.6e}, std={np.std(propagation_distances):.6e}"
            )

        ptychogram = Ptychogram(
            diffraction_patterns=np.array(predicted_diffraction_patterns),
            propagation_distance=propagation_distances,  # Per-position array
            sample_positions=np.array(interaction_coordinates.translation),
            sample_orientations=np.array(interaction_coordinates.rotation._representation_6d),
            detector_positions=np.array(detector_coordinates.translation),
            detector_orientations=np.array(detector_coordinates.rotation._representation_6d),
            pixel_size=np.array(self.detector.sampling.pixel_size),
            wavelength=np.array(self.illumination.probe.wavelength),
            detector_darkframe=np.array(self.detector.dark_counts),
            loaded_from="SIMULATION",
        )
        return ptychogram

    @staticmethod
    def initialize_sample_coordinates(ptychogram: Ptychogram) -> CoordinateSystem:
        """Create a `CoordinateSystem` for sample positions and orientations
        from a `Ptychogram`. The sample translations are normalized for better
        optimization performance. Output is wrapped in
        `IndexSliceParametrization` to specify indexing over the dataset
        dimension.

        Args:
            ptychogram: Source dataset containing `sample_orientations` and `sample_positions`.

        Returns:
            A `CoordinateSystem` with normalized translations suitable for initializing interactions.
        """
        sample_orientations = Rotation(jnp.asarray(ptychogram.sample_orientations))
        translation = jnp.asarray(ptychogram.sample_positions)
        sample_translation_scale = np.max(np.linalg.norm(ptychogram.sample_positions, axis=-1))

        sample_translations = NormalizedReferencedArrayParametrization(translation, scale=sample_translation_scale)
        sample_translations = sample_translations
        sample_coordinates = IndexSliceParameter(
            CoordinateSystem(
                sample_orientations,
                sample_translations,
                model_in_local_frame=True,
            )
        )

        return sample_coordinates

    @staticmethod
    def initialize_detector_coordinates(ptychogram: Ptychogram) -> CoordinateSystem:
        """Create a `CoordinateSystem` for detector positions and orientations
        from a `Ptychogram`.

        The detector translations are normalized for better optimization performance. Output is wrapped in
        `IndexSliceParametrization` to specify indexing over the dataset dimension.

        Args:
            ptychogram: Source dataset containing `detector_orientations` and `detector_positions`.

        Returns:
            A `CoordinateSystem` normalized for detector geometry initialization.
        """
        detector_orientations = Rotation(jnp.asarray(ptychogram.detector_orientations))
        translation = jnp.asarray(ptychogram.detector_positions)
        detector_translation_scale = np.mean(np.linalg.norm(translation, axis=-1))

        detector_translations = NormalizedReferencedArrayParametrization(translation, scale=detector_translation_scale)
        detector_translations = detector_translations
        detector_coordinates = IndexSliceParameter(
            CoordinateSystem(
                detector_orientations,
                detector_translations,
            )
        )
        return detector_coordinates

    @eqx.filter_jit
    def __call__(self, **kwargs) -> tuple[Float[Array, "* m n"], Bool[Array, "* d"]]:
        """Predict images from the model.

        This will be called inside the jit-boundary, so all operations contained herein must be jittable.
        The output must be a single image of shape (m, n). Before this function is called, any parts of
        the model which are types of parametrizations will be evaluated to their underlying arrays.
        Therefore, inside of this function, all parametrizations can be treated as normal arrays.
        For example, any model parameters which are instances of IndexSliceParametrization (with leading
        dimension d for the dataset index), will have their leading dimension removed inside of
        :py:meth:`~ptyrax.models.ImagePredictionModel.__call__`. This way, the model needs not worry about
        including batch dimensions at all, every prediction should be just for a single image.
        """

        @eqx.filter_vmap
        def coherent_forward_model(illumination: IlluminationModel) -> tuple[CoherentField, Bool[Array, " d"]]:
            probe = illumination()
            exit_field = self.interaction(probe)
            detector_coordinates = self.detector.coordinates.at_current_index()
            propagated_field, _ = eqx.filter_jit(self.propagator)(
                exit_field, detector_coordinates, self.detector.sampling
            )
            return propagated_field

        propagated_fields = coherent_forward_model(self.illumination)
        detected_fields = self.detector(propagated_fields)

        return detected_fields

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.detector.sampling.shape

    def __regularize__(self) -> tuple[float, list[NamedLoss]]:
        all_regularizations = []
        all_total = 0.0
        for name, attr in self.__dict__.items():
            if hasattr(attr, "__regularize__"):
                total, regularizations = attr.__regularize__()
                if regularizations == []:
                    continue
                regularizations = [NamedLoss(f"{name}.{r.tag}", r.value) for r in regularizations]
                all_regularizations.extend(regularizations)
                all_total += total
        return all_total, all_regularizations

    def __log_epoch__(self, writer: SummaryWriter, epoch: int, **kwargs) -> None:
        for attr in self.__dict__.values():
            if hasattr(attr, "__log_epoch__"):
                attr.__log_epoch__(writer, epoch, **kwargs)

    def __plot__(self, *args, **kwargs) -> None:
        """Display the geometry of the ptychography setup.

        including the detector and illumination directions on a unit
        sphere, as well as the Fourier scattering vectors in the sample
        frame. This is useful for visualizing the experimental geometry
        and understanding how the sample is being probed. The function
        computes the necessary geometric quantities and creates three
        subplots: one showing the detector and illumination directions
        on a unit sphere in the XZ plane, one showing the Fourier
        scattering vectors in the XY plane, and one showing the overall
        geometry of the sample and detector.
        """
        self = resolve_parametrizations(self)
        edges = self.detector.sampling.edges()
        detector_rotation_matrix = self.detector.coordinates.rotation.as_matrix()

        detector_coordinate_edges = self.detector.coordinates.translation[:, jnp.newaxis, :] + jnp.einsum(
            "sdj, nj -> snd",
            detector_rotation_matrix.transpose((0, 2, 1)),
            edges,
        )

        detector_coordinate_sphere = detector_coordinate_edges / jnp.linalg.norm(
            detector_coordinate_edges, axis=-1, keepdims=True
        )

        illumination_edges = edges[:, jnp.newaxis, :] + jnp.linalg.norm(
            self.detector.coordinates.translation, axis=-1, keepdims=True
        ) * jnp.array([[0.0, 0.0, 1.0]])
        illumination_edges = illumination_edges.transpose((1, 0, 2))
        illumination_coordinate_sphere = illumination_edges / jnp.linalg.norm(
            illumination_edges, axis=-1, keepdims=True
        )

        scattering_vector = (
            detector_coordinate_sphere[:, :, jnp.newaxis, :] - illumination_coordinate_sphere[:, jnp.newaxis, :, :]
        )

        sample_rotation_matrix = self.interaction.coordinates.rotation.as_matrix()
        fig = plt.figure(figsize=(10, 5), layout="constrained")
        gs = fig.add_gridspec(1, 3)
        fig, ax1 = plot_spheres_xz(
            sample_rotation_matrix,
            detector_coordinate_sphere,
            illumination_coordinate_sphere,
            scattering_vector,
            fig=fig,
            gs=gs[0, 1],
        )
        ax1.set_title("Fourier scattering vectors in sample frame (XZ plane)", fontdict={"fontsize": 10})

        fig, ax2 = plot_scattering_vector_xy(
            scattering_vector,
            sample_rotation_matrix,
            fig=fig,
            gs=gs[0, 2],
        )
        ax2.set_title("Fourier scattering vectors in sample frame (XY plane)", fontdict={"fontsize": 10})

        fig, ax3 = plot_geometry(
            self.interaction.coordinates,
            self.detector.coordinates,
            detector_coordinate_edges,
            fig=fig,
            gs=gs[0, 0],
        )
        ax3.set_title("Sample and detector geometry", fontdict={"fontsize": 10})
        plt.show()
        plt.close(fig)


@gin.configurable()
def initialize_3d_tilted_sampling(
    detector_sampling: SamplingGrid,
    shape: tuple[int, int],
    sample_coordinates: CoordinateSystem,
    detector_coordinates: CoordinateSystem,
    wavelengths: Array,
    n_dim: Union[Literal[2], Literal[3]] = 2,
    fourier_oversampling_factor: np.ndarray = np.array([1.0, 1.0]),
    real_oversampling_factor: np.ndarray = np.array([1.0, 1.0]),
    probe_fourier_oversampling_factor: np.ndarray = None,
    writer: Optional["SummaryWriter"] = None,
    epoch: Optional[int] = None,
    prefix: str = "",
) -> tuple[SamplingGrid, SamplingGrid, SamplingGrid]:
    r"""Compute interaction, probe, and forward sampling grids from the
    scattering geometry.

    Given the detector and sample coordinate systems, this function determines
    the Fourier bounds of the scattering vectors in the sample frame and derives
    real-space pixel sizes that satisfy the Nyquist condition for the tilted
    geometry. It returns three :py:class:`~ptyrax.spatial.SamplingGrid` instances
    defining the discretization for the interaction (sample), probe, and forward
    propagation fields. For most use cases, the probe and forward sampling grids
    will be the same, but separate oversampling factors can be provided for flexibility.

    The real-space pixel size is computed as:

    .. math::

        \Delta x = \frac{\lambda_{\min}}{\xi_{\max}}

    where $\xi_{\max}$ is the maximum angular frequency extent in the sample frame
    and $\lambda_{\min}$ is the shortest wavelength.

    Args:
        detector_sampling: Pixel grid of the detector (shape and pixel size).
        shape: Nominal detector shape ``(nx, ny)`` used as base for grid dimensions.
        sample_coordinates: Sample positions and orientations (may be wrapped in
            :py:class:`~ptyrax.parametrizations.IndexSliceParameter`).
        detector_coordinates: Detector positions and orientations.
        wavelengths: Array of illumination wavelengths.
        n_dim: Dimensionality of the scattering geometry, 2 (planar) or 3 (full 3-D
            Fourier bounds including $\xi_z$).
        fourier_oversampling_factor: Multiplicative factor(s) applied to the Fourier
            bounds for the forward grid.
        real_oversampling_factor: Multiplicative factor(s) for the number of grid
            points in the forward grid.
        probe_fourier_oversampling_factor: If provided, separate oversampling
            factor(s) for the probe grid (otherwise matches forward grid).
        writer: Optional TensorBoard writer for logging geometry plots.
        epoch: Epoch number for TensorBoard logging.
        prefix: Prefix string for TensorBoard tags.

    Returns:
        A tuple ``(interaction_sampling, probe_sampling, forward_sampling)`` of
        :py:class:`~ptyrax.spatial.SamplingGrid` instances.
    """
    fourier_oversampling_factor = make_length_n(fourier_oversampling_factor, 2)
    real_oversampling_factor = make_length_n(real_oversampling_factor, 2)
    detector_coordinates = resolve_parametrizations(detector_coordinates)
    sample_coordinates = resolve_parametrizations(sample_coordinates)
    shape_array = jnp.asarray(shape)

    (
        detector_coordinate_edges,
        detector_coordinate_sphere,
        illumination_coordinate_sphere,
        scattering_vector,
        sample_rotation_matrix,
        sample_frame_angular_coordinates,
    ) = _compute_scattering_geometry(detector_sampling, detector_coordinates, sample_coordinates)

    fourier_bounds, interaction_modulation_height, adjusted_angular_coords = _compute_fourier_bounds(
        sample_frame_angular_coordinates, n_dim
    )

    forward_sampling, forward_fourier_bounds = _compute_forward_sampling(
        fourier_bounds, fourier_oversampling_factor, real_oversampling_factor, shape_array, wavelengths
    )

    probe_sampling = _compute_probe_sampling(
        probe_fourier_oversampling_factor,
        fourier_bounds,
        forward_sampling,
        forward_fourier_bounds,
        real_oversampling_factor,
        shape_array,
        wavelengths,
    )

    _log_or_show_sampling_geometry(
        writer,
        epoch,
        sample_rotation_matrix,
        detector_coordinate_sphere,
        illumination_coordinate_sphere,
        scattering_vector,
        sample_coordinates,
        detector_coordinates,
        detector_coordinate_edges,
    )

    interaction_sampling = _compute_interaction_sampling(
        sample_coordinates,
        sample_rotation_matrix,
        forward_sampling,
        probe_sampling,
        shape_array,
        interaction_modulation_height,
    )
    return interaction_sampling, probe_sampling, forward_sampling


def _compute_scattering_geometry(
    detector_sampling: SamplingGrid,
    detector_coordinates: CoordinateSystem,
    sample_coordinates: CoordinateSystem,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Compute detector/illumination spheres and scattering vectors in sample
    frame."""
    edges = detector_sampling.edges()
    detector_rotation_matrix = detector_coordinates.rotation.as_matrix()

    detector_coordinate_edges = detector_coordinates.translation[:, jnp.newaxis, :] + jnp.einsum(
        "sdj, nj -> snd",
        detector_rotation_matrix.transpose((0, 2, 1)),
        edges,
    )

    detector_coordinate_sphere = detector_coordinate_edges / jnp.linalg.norm(
        detector_coordinate_edges, axis=-1, keepdims=True
    )

    illumination_edges = edges[:, jnp.newaxis, :] + jnp.linalg.norm(
        detector_coordinates.translation, axis=-1, keepdims=True
    ) * jnp.array([[0.0, 0.0, 1.0]])
    illumination_edges = illumination_edges.transpose((1, 0, 2))
    illumination_coordinate_sphere = illumination_edges / jnp.linalg.norm(illumination_edges, axis=-1, keepdims=True)

    scattering_vector = (
        detector_coordinate_sphere[:, :, jnp.newaxis, :] - illumination_coordinate_sphere[:, jnp.newaxis, :, :]
    )

    sample_rotation_matrix = sample_coordinates.rotation.as_matrix()
    sample_frame_angular_coordinates = jnp.einsum("sid, snmd -> snmi", sample_rotation_matrix, scattering_vector)
    return (
        detector_coordinate_edges,
        detector_coordinate_sphere,
        illumination_coordinate_sphere,
        scattering_vector,
        sample_rotation_matrix,
        sample_frame_angular_coordinates,
    )


def _compute_fourier_bounds(
    sample_frame_angular_coordinates: Array,
    n_dim: Union[Literal[2], Literal[3]],
) -> tuple[Array, Array, Array]:
    """Compute Fourier bounds and modulation height given angular
    coordinates."""
    if n_dim == 2:
        fourier_bounds = jnp.amax(jnp.abs(sample_frame_angular_coordinates), axis=(0, 1, 2))[:2]
        interaction_modulation_height = jnp.array([0.0, 0.0])
        adjusted = sample_frame_angular_coordinates
    elif n_dim == 3:
        mins = jnp.min(sample_frame_angular_coordinates, axis=(0, 1, 2))
        maxes = jnp.max(sample_frame_angular_coordinates, axis=(0, 1, 2))
        interaction_modulation_height = (maxes + mins) / 2
        adjusted = sample_frame_angular_coordinates - jnp.array([0.0, 0.0, interaction_modulation_height])
        fourier_bounds = jnp.amax(jnp.abs(adjusted), axis=(0, 1, 2))
    else:
        raise ValueError("The shape of the sampling grid must be 2D or 3D")
    return fourier_bounds, interaction_modulation_height, adjusted


def _compute_forward_sampling(
    fourier_bounds: Array,
    fourier_oversampling_factor: np.ndarray,
    real_oversampling_factor: np.ndarray,
    shape: Array,
    wavelengths: Array,
) -> tuple[SamplingGrid, Array]:
    """Compute forward (interaction) Fourier bounds and sampling grid."""
    forward_fourier_bounds = fourier_bounds * np.array(fourier_oversampling_factor)
    forward_shape = shape * np.array(fourier_oversampling_factor) * np.array(real_oversampling_factor)
    forward_real_space_pixel_size = 1 / (forward_fourier_bounds) * np.min(wavelengths)

    logging.info(f"fourier_bounds={fourier_bounds}")
    logging.info(f"forward_fourier_bounds={forward_fourier_bounds}")
    logging.info(f"forward_real_space_pixel_size={forward_real_space_pixel_size}")
    pixel_anisotropy = np.abs(forward_real_space_pixel_size[0] - forward_real_space_pixel_size[1])
    logging.info(f"Pixel size anisotropy: {pixel_anisotropy}")

    shape_tuple = (int(forward_shape[0]), int(forward_shape[1]))
    forward_sampling = SamplingGrid.from_tuples(shape_tuple, forward_real_space_pixel_size)
    return forward_sampling, forward_fourier_bounds


def _compute_probe_sampling(
    probe_fourier_oversampling_factor: np.ndarray | None,
    fourier_bounds: Array,
    forward_sampling: SamplingGrid,
    forward_fourier_bounds: Array,
    real_oversampling_factor: np.ndarray,
    shape: Array,
    wavelengths: Array,
) -> SamplingGrid:
    """Compute probe sampling grid, potentially with different oversampling."""
    if probe_fourier_oversampling_factor is None:
        return forward_sampling

    probe_fourier_oversampling_factor = make_length_n(probe_fourier_oversampling_factor, 2)
    probe_fourier_bounds = fourier_bounds * np.array(probe_fourier_oversampling_factor)
    probe_shape = shape * np.array(real_oversampling_factor) * np.array(probe_fourier_oversampling_factor)
    probe_real_space_pixel_size = 1 / (probe_fourier_bounds) * np.min(wavelengths)

    shape_tuple = (int(probe_shape[0]), int(probe_shape[1]))
    probe_sampling = SamplingGrid.from_tuples(shape_tuple, probe_real_space_pixel_size)
    return probe_sampling


def _log_or_show_sampling_geometry(
    writer: Optional["SummaryWriter"],
    epoch: Optional[int],
    sample_rotation_matrix: Array,
    detector_coordinate_sphere: Array,
    illumination_coordinate_sphere: Array,
    scattering_vector: Array,
    sample_coordinates: CoordinateSystem,
    detector_coordinates: CoordinateSystem,
    detector_coordinate_edges: Array,
    show_plot: bool = False,
    log_plot: bool = True,
) -> None:
    if not log_plot and not show_plot:
        return
    """Plot and either log to TensorBoard or show sampling geometry."""

    fig1, _ = plot_spheres_xz(
        sample_rotation_matrix, detector_coordinate_sphere, illumination_coordinate_sphere, scattering_vector
    )
    if writer is not None and log_plot:
        writer.add_figure("4_model/model_sampling/spheres_xz", fig1, epoch if epoch is not None else 0)
    if show_plot:
        plt.show()
    plt.close(fig1)

    fig2, _ = plot_scattering_vector_xy(scattering_vector, sample_rotation_matrix)
    if writer is not None and log_plot:
        writer.add_figure("4_model/model_sampling/scattering_vector_xy", fig2, epoch if epoch is not None else 0)
    if show_plot:
        plt.show()
    plt.close(fig2)

    fig3, _ = plot_geometry(sample_coordinates, detector_coordinates, detector_coordinate_edges)
    if writer is not None and log_plot:
        writer.add_figure("4_model/model_sampling/geometry", fig3, epoch if epoch is not None else 0)
    if show_plot:
        plt.show()

    plt.close(fig3)


def _compute_interaction_sampling(
    sample_coordinates: CoordinateSystem,
    sample_rotation_matrix: Array,
    forward_sampling: SamplingGrid,
    probe_sampling: SamplingGrid,
    shape: Array,
    interaction_modulation_height: Array,
) -> SamplingGrid:
    """Compute interaction sampling grid including shift padding."""
    sample_shift = jnp.einsum("ndi, ni -> nd", sample_rotation_matrix, sample_coordinates.translation)
    sample_shift = sample_shift[:, :2]
    shift_bounds = np.max(jnp.abs(sample_shift), axis=0) / forward_sampling.pixel_size

    if np.min(shift_bounds) < 2:
        logging.warning(
            "The ptychogram shifts are much smaller than the shape of the diffraction pattern. "
            "Does the ptychogram have the correct units?"
        )

    if np.max(shift_bounds) > 10 * np.max(shape):
        logging.warning(
            "The ptychogram shifts are much larger than the field of view of a single diffraction pattern. "
            "Does the ptychogram have the correct units?"
        )

    shift_bounds = np.ceil(shift_bounds)
    logging.info(f"{shift_bounds=}")
    logging.info(f"{probe_sampling.pixel_size=}")
    new_shape = np.array(forward_sampling.shape, dtype=int) + 2 * shift_bounds.astype(int)

    return SamplingGrid.from_tuples(
        (int(new_shape[0]), int(new_shape[1])),
        (forward_sampling.pixel_size[0], forward_sampling.pixel_size[1]),
        interaction_modulation_height,
    )


@gin.configurable
def plot_model(
    model: PtychographyModel,
    show: bool = True,
    fig: Figure = None,
    ax: Axes = None,
    **kwargs,
) -> tuple[Figure, SubplotSpec, list[AxesImage]]:
    """Plot the probe and illumination fields of a
    :py:class:`PtychographyModel`.

    Displays the model's probe field and illumination state side by side.
    This is a gin-configurable convenience function for quick visual inspection
    of the current model state during reconstruction.

    Args:
        model: The ptychography model to visualize.
        show: Whether to call ``plt.show()`` after plotting.
        fig: Existing figure to draw into. If ``None``, a new figure is created.
        ax: Axes array with at least two elements. If ``None``, new axes are created.
        **kwargs: Additional keyword arguments forwarded to
            :py:func:`~ptyrax.utils.plot`.

    Returns:
        A tuple of ``(figure, axes, images)`` where ``images`` is the combined
        list of :class:`~matplotlib.image.AxesImage` objects from both subplots.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 2)
    try:
        _, _, ims_1 = plot(model.illumination.probe, ax=ax[0], show=False, **kwargs)
        _, _, ims_2 = plot(model.illumination, ax=ax[1], show=show, **kwargs)
    except TypeError:
        logging.warning(
            "Could not use specified axes to (must have at least two axes). Continuing with a new figure..."
        )
        return plot_model(model, show=show, fig=None, ax=None, **kwargs)
    return fig, ax, ims_1 + ims_2


# endregion
# region Model Adjusters


@gin.configurable()
def scale_illumination_equal_pixel_size(
    model: PtychographyModel,
    scale: float = 1.0,
) -> PtychographyModel:
    """Rescale the probe illumination field by interpolating to a new pixel
    size.

    Multiplies the probe's pixel size by ``scale`` and interpolates the probe data
    onto the new grid, preserving the field of view in pixels while changing the
    physical extent. This is typically used when the wavelength is modified and the
    probe must be adjusted to maintain consistent Fourier sampling.

    Args:
        model: The ptychography model whose illumination will be rescaled.
        scale: Multiplicative factor for the pixel size. Values > 1 enlarge pixels
            (zoom out), values < 1 shrink pixels (zoom in).

    Returns:
        A new :py:class:`PtychographyModel` with the rescaled probe data.
    """
    if not jnp.isscalar(scale):
        return scale_illumination_equal_pixel_size(model, scale=scale[0])
    probe_data = model.illumination.probe()[..., 0]
    if probe_data.ndim == 2:
        probe_data = probe_data[None, ...]
    probe_sampling = model.illumination.probe.sampling
    probe_sampling_single = probe_sampling[0] if jnp.asarray(probe_sampling.pixel_size).ndim > 1 else probe_sampling
    target_idx = (0, 0)
    original_pixel_size = jnp.array((probe_sampling_single.x_pixel_size, probe_sampling_single.y_pixel_size))
    target_pixel_size = original_pixel_size * scale
    new_probe_data = shift_with_interpolation_unequal_pixel_size(
        probe_data,
        original_pixel_size,
        target_idx,
        probe_sampling_single.shape,
        target_pixel_size,
    )
    new_probe_data = new_probe_data[..., jnp.newaxis]
    new_model = set_probe_data_preserve_parametrization(model, new_probe_data)
    return new_model


@gin.configurable
def scale_model_wavelength(
    model: PtychographyModel,
    scale: float = 1.0,
    rescale_illumination: bool = True,
    **kwargs,
) -> PtychographyModel:
    """Scale the model wavelength by a multiplicative factor.

    Multiplies all wavelength entries by ``scale``. Optionally rescales the
    illumination probe pixel size to maintain consistent Fourier sampling after
    the wavelength change.

    Args:
        model: The ptychography model to modify.
        scale: Multiplicative factor applied to the wavelength.
        rescale_illumination: If ``True``, also rescales the probe via
            :py:func:`scale_illumination_equal_pixel_size` to compensate for the
            wavelength change.

    Returns:
        A new :py:class:`PtychographyModel` with the scaled wavelength.
    """
    wavelength = model.illumination.probe.wavelength
    new_wavelength = wavelength * scale
    new_model = eqx.tree_at(lambda m: m.illumination.probe.wavelength, model, new_wavelength)
    if rescale_illumination:
        new_model = scale_illumination_equal_pixel_size(new_model, scale=scale)
    return new_model


@gin.configurable
def set_model_wavelength(
    model: PtychographyModel,
    wavelength: float = 1.0,
    rescale_illumination: bool = True,
    **kwargs,
) -> PtychographyModel:
    """Set the model wavelength to an absolute value.

    Replaces the current wavelength(s) with the given value (broadcast to match
    shape). Optionally rescales the illumination probe to maintain consistent
    Fourier sampling at the new wavelength.

    Args:
        model: The ptychography model to modify.
        wavelength: The new wavelength value in the same units as the model.
        rescale_illumination: If ``True``, rescales the probe via
            :py:func:`scale_illumination_equal_pixel_size` to compensate.

    Returns:
        A new :py:class:`PtychographyModel` with the updated wavelength.
    """
    old_wavelength = model.illumination.probe.wavelength
    # Broadcasting hack to ensure shape matches
    new_wavelength = old_wavelength * 0 + wavelength
    new_model = eqx.tree_at(lambda m: m.illumination.probe.wavelength, model, new_wavelength)
    if rescale_illumination:
        new_model = scale_illumination_equal_pixel_size(new_model, scale=old_wavelength / new_wavelength)
    return new_model


@gin.configurable
def set_model_constant_tilt_angle(
    model: PtychographyModel,
    tilt_angle: float = 0.0,  # deg
    detector_tilt_angle: float | None = None,
    **kwargs,
) -> PtychographyModel:
    """Override sample and detector orientations with those corresponding to a
    constant tilt angle.

    Sets all sample orientations to a uniform rotation about the y-axis by
    ``tilt_angle`` degrees, recomputes sample positions in the new frame, and
    derives the specular detector orientation and position. An independent
    ``detector_tilt_angle`` can be specified when the detector does not sit
    exactly at the specular reflection.

    Args:
        model: The ptychography model to modify.
        tilt_angle: Sample tilt angle in degrees (rotation about y-axis).
        detector_tilt_angle: If provided, overrides the detector orientation
            independently from the sample tilt.

    Returns:
        A new :py:class:`PtychographyModel` with updated coordinate systems.
    """
    original_model = model
    model = resolve_parametrizations(model)
    tilt_angle = jnp.array([0, tilt_angle, 0])

    def sample_orientation_from_tilt_angle(tilt_angle: float) -> tuple[Float[Array, " n 6"], Float[Array, " n 3"]]:
        tilt_angle = jnp.array(tilt_angle)
        local_frame_sample_positions = model.interaction.coordinates.translation_internal
        rotation_matrix = JaxRotation.from_euler("xyz", tilt_angle, degrees=True).as_matrix()
        new_sample_orientations = jnp.tile(
            matrix_to_six_dimensional_representation(rotation_matrix),
            (len(local_frame_sample_positions), 1),
        )
        return new_sample_orientations, local_frame_sample_positions

    new_sample_orientations, local_frame_sample_positions = sample_orientation_from_tilt_angle(tilt_angle)
    new_sample_positions = jnp.einsum(
        "ndi, ni -> nd",
        # transpose = inverse: From local to global
        six_dimensional_representation_to_matrix(new_sample_orientations).transpose((0, 2, 1)),
        local_frame_sample_positions,
    )

    new_detector_orientations = matrix_to_six_dimensional_representation(
        R_y(180)
        @ six_dimensional_representation_to_matrix(new_sample_orientations)
        @ six_dimensional_representation_to_matrix(new_sample_orientations)
    )
    propagation_distances = jnp.linalg.norm(model.detector.coordinates.translation, axis=-1)
    new_detector_positions = jnp.einsum(
        "ndi, ni -> nd",
        six_dimensional_representation_to_matrix(new_detector_orientations).transpose((0, 2, 1)),
        jnp.stack((0 * propagation_distances, 0 * propagation_distances, propagation_distances), axis=-1),
    )
    if detector_tilt_angle is not None:
        new_sample_orientations, _ = sample_orientation_from_tilt_angle(detector_tilt_angle)
        new_detector_orientations = matrix_to_six_dimensional_representation(
            R_y(180)
            @ six_dimensional_representation_to_matrix(new_sample_orientations)
            @ six_dimensional_representation_to_matrix(new_sample_orientations)
        )

    new_model = eqx.tree_at(
        lambda m: m.interaction.coordinates,
        original_model,
        IndexSliceParameter(
            CoordinateSystem(
                Rotation(new_sample_orientations),
                new_sample_positions,
                normalize_translation=False,
                model_in_local_frame=model.interaction.coordinates.model_in_local_frame,
            )
        ),
    )

    new_model = eqx.tree_at(
        lambda m: m.detector.coordinates,
        new_model,
        IndexSliceParameter(
            CoordinateSystem(
                Rotation(new_detector_orientations),
                new_detector_positions,
                normalize_translation=False,
                model_in_local_frame=model.interaction.coordinates.model_in_local_frame,
            )
        ),
    )
    return new_model


@gin.configurable
def set_outside_scan_range_to(
    model: PtychographyModel,
    epoch: int,
    optimizer_state: PyTree[PtychographyModel],
    extra_range_factor: Union[tuple[float], float] = 1.0,
    set_to: Union[Callable, ScalarLike] = jnp.mean,
    deviation_scale: float = 0.2,
    apply_every: int = 1,
) -> PtychographyModel:
    """Suppress reconstruction artifacts outside the scanned area.

    Replaces the interaction (reflection coefficient) values at positions
    beyond the scan range with a constant or averaged value. A soft
    suppression term further damps deviations from the replacement value
    to prevent edge artifacts from growing during optimization.

    Args:
        model: The ptychography model to modify.
        epoch: Current optimization epoch (used with ``apply_every``).
        optimizer_state: Current optimizer state (unused but required by
            the adjuster signature).
        extra_range_factor: Multiplicative factor applied to scan displacements
            when computing the boundary radius.
        set_to: Value or callable to compute the replacement value. If callable,
            it is applied to the full coefficient array (e.g. ``jnp.mean``).
        deviation_scale: Fraction controlling how strongly existing deviations
            outside the boundary are suppressed.
        apply_every: Only apply the adjustment every ``apply_every`` epochs.

    Returns:
        A new :py:class:`PtychographyModel` with suppressed out-of-range values.
    """
    if epoch % apply_every != 0:
        return model
    resolved_model = resolve_parametrizations(model)
    translations = resolved_model.interaction.coordinates.translation_internal
    translations_xy = translations[:, :2]
    displacements = translations_xy - jnp.mean(translations_xy, axis=0)
    displacements = displacements * (jnp.array(extra_range_factor)[jnp.newaxis])
    max_distance = jnp.max(jnp.linalg.norm(displacements, axis=-1))

    xx, yy = model.interaction.sampling.meshgrid
    rr = jnp.sqrt(xx**2 + yy**2)

    if not isinstance(model.interaction.reflection_coefficient, (DirectArrayParametrization, jnp.ndarray)):
        raise TypeError(
            "Can only adjust model positions if they are modeled as DirectArrayParametrizations."
            f"Got {model.interaction.reflection_coefficient}"
        )

    coefficients = (
        model.interaction.reflection_coefficient()
        if isinstance(model.interaction.reflection_coefficient, DirectArrayParametrization)
        else model.interaction.reflection_coefficient
    )
    if callable(set_to):
        replacement_value = set_to(coefficients)
    else:
        replacement_value = jnp.complex64(set_to)
    replacement_coefficients = (rr > max_distance) * replacement_value
    # Further suppress artifacts by raising the reflectance wherever the optimization made changes
    deviations = coefficients - replacement_coefficients
    new_coefficients = jnp.where(
        rr > max_distance, replacement_coefficients - deviation_scale * deviations, coefficients
    )
    new_coefficients = (
        type(model.interaction.reflection_coefficient)(new_coefficients)
        if isinstance(model.interaction.reflection_coefficient, DirectArrayParametrization)
        else new_coefficients
    )

    new_model = eqx.tree_at(lambda m: m.interaction.reflection_coefficient, model, new_coefficients)
    return new_model


@gin.configurable
def replace_interaction(
    model: PtychographyModel,
    new_interaction_generator: Callable[[IlluminationModel], IlluminationModel]
    | list[Callable[[IlluminationModel], IlluminationModel]] = identity,
    **kwargs,
) -> PtychographyModel:
    """Replace the model's interaction with a transformed version.

    Applies one or more generator functions sequentially to the current
    interaction model and replaces it in the ptychography model. This is
    the primary mechanism for swapping interaction types or applying
    structural changes (e.g. converting to multi-slice).

    Args:
        model: The ptychography model to modify.
        new_interaction_generator: A callable or list of callables, each
            taking an :py:class:`~ptyrax.models.interaction.InteractionModel`
            and returning a new one.

    Returns:
        A new :py:class:`PtychographyModel` with the replaced interaction.
    """
    new_interaction = model.interaction
    if isinstance(new_interaction_generator, list):
        for gen in new_interaction_generator:
            if gen is None:
                continue
            new_interaction = gen(new_interaction)
    elif new_interaction_generator is None:
        return model
    else:
        new_interaction = new_interaction_generator(model.interaction)
    new_model = eqx.tree_at(lambda m: m.interaction, model, new_interaction)
    return new_model


@gin.configurable
def set_interaction_real_only(
    model: PtychographyModel,
    **kwargs,
) -> PtychographyModel:
    """Discard the imaginary part of the interaction reflection coefficient.

    Replaces the complex reflection coefficient with its real part only.
    This is useful for enforcing a purely absorptive (no phase) sample model
    or for resetting phase artifacts.

    Args:
        model: The ptychography model to modify.

    Returns:
        A new :py:class:`PtychographyModel` with a real-valued reflection coefficient.
    """
    coefficients = (
        model.interaction.reflection_coefficient()
        if isinstance(model.interaction.reflection_coefficient, ArrayParametrization)
        else model.interaction.reflection_coefficient
    )
    new_coefficients = jnp.real(coefficients)
    new_coefficients = (
        type(model.interaction.reflection_coefficient)(new_coefficients)
        if isinstance(model.interaction.reflection_coefficient, ArrayParametrization)
        else new_coefficients
    )
    new_model = eqx.tree_at(lambda m: m.interaction.reflection_coefficient, model, new_coefficients)
    return new_model


@gin.configurable
def replace_probe(
    model: PtychographyModel,
    new_probe_generator: Callable[[CoherentField], CoherentField] | list[Callable[[CoherentField], CoherentField]],
) -> PtychographyModel:
    """Replace the model's probe field with a transformed version.

    Applies one or more generator functions sequentially to the current probe
    (:py:class:`~ptyrax.field.CoherentField`) and updates the model. Only works
    with :py:class:`~ptyrax.models.illumination.DirectIllumination`.

    Args:
        model: The ptychography model to modify.
        new_probe_generator: A callable or list of callables, each taking a
            :py:class:`~ptyrax.field.CoherentField` and returning a new one.

    Returns:
        A new :py:class:`PtychographyModel` with the replaced probe.

    Raises:
        TypeError: If the illumination model is not
            :py:class:`~ptyrax.models.illumination.DirectIllumination`.
    """
    if not isinstance(model.illumination, DirectIllumination):
        raise TypeError(
            "Can only replace probe if the illumination is modeled as DirectIllumination."
            f"Got {type(model.illumination)}"
        )
    if not isinstance(new_probe_generator, list | tuple):
        new_probe_generator = (new_probe_generator,)
    new_probe = model.illumination.probe
    for gen in new_probe_generator:
        new_probe = gen(new_probe)
    new_model = eqx.tree_at(lambda m: m.illumination.probe, model, new_probe)
    return new_model


@gin.configurable
def replace_interaction_from_hdf5(
    model: PtychographyModel,
    hdf5_path: str,
    hdf5_interaction_path: str = "interaction",
    **kwargs,
) -> PtychographyModel:
    """Replace the model's interaction parameters with values loaded from an
    HDF5 file.

    Loads previously saved interaction model state from an HDF5 reconstruction
    file and applies it to the current model's interaction subtree.

    Args:
        model: The ptychography model to modify.
        hdf5_path: Path to the HDF5 file containing saved interaction parameters.
        hdf5_interaction_path: Group path prefix within the HDF5 file where
            the interaction parameters are stored.
        **kwargs: Additional keyword arguments forwarded to
            :py:func:`~ptyrax.hdf5_checkpoint.apply_hdf5_to_model`.

    Returns:
        A new :py:class:`PtychographyModel` with interaction loaded from HDF5.
    """
    new_interaction, _, _ = apply_hdf5_to_model(
        model.interaction,
        hdf5_path,
        path_prefix=hdf5_interaction_path,
        **kwargs,
    )

    new_model = eqx.tree_at(lambda m: m.interaction, model, new_interaction)
    return new_model


@gin.configurable
def multiply_interaction_xiz_function(
    model: PtychographyModel,
    tilt_angle: float = 0.0,
    thickness: float = 0.0,
    xiz_function: Callable[[Inexact[Array, "m n"]], Inexact[Array, "m n"]] = jnp.cos,
) -> PtychographyModel:
    r"""Multiply the interaction reflection coefficient by a depth-dependent
    transfer function.

    Applies a zero-order correction for sample depth by multiplying the
    reflection coefficient in Fourier space by a function of the out-of-plane
    spatial frequency $\xi_z$. The transfer function models the effect of
    finite sample thickness on the scattered wave.

    The $\xi_z$ component is computed from the tilt geometry as:

    .. math::

        \xi_z = k \left( \sqrt{1 - (\xi_x + s_x)^2 - (\xi_y + s_y)^2} - s_z \right)

    where $k = 2\pi / \lambda$ and $(s_x, s_y, s_z)$ is the specular direction.

    Args:
        model: The ptychography model to modify.
        tilt_angle: Sample tilt angle in degrees, used to compute the specular
            direction.
        thickness: Physical thickness of the sample layer. Controls the argument
            to ``xiz_function`` as ``thickness / 2 * xi_z``.
        xiz_function: Function applied element-wise to the scaled $\xi_z$ array.
            Common choices are ``jnp.cos`` (default, for two-layer models)
            or ``jnp.sinc`` (for rectangular models).

    Returns:
        A new :py:class:`PtychographyModel` with the modified interaction.

    Raises:
        TypeError: If the interaction is not a
            :py:class:`~ptyrax.models.interaction.FresnelReflection`.
    """
    interaction = model.interaction
    wavelength = model.illumination().wavelength
    if not isinstance(interaction, FresnelReflection):
        raise TypeError(
            f"Can only apply xiz_function if the interaction is modeled as FresnelReflection.Got {type(interaction)}"
        )
    xi_x, xi_y = interaction.sampling.to_far_field(jnp.max(wavelength), 1.0).meshgrid
    sx, sy, sz = (jnp.sin(np.deg2rad(tilt_angle)), 0, -np.cos(np.deg2rad(tilt_angle)))
    assert jnp.isclose(jnp.sqrt(1 - (sx**2 + sy**2)) - sz, 2 * jnp.cos(np.deg2rad(tilt_angle))), (  # noqa: S101
        "Tilt angle is not consistent with the provided sx, sy, sz values. Please check the calculations."
    )
    k = 2 * np.pi / wavelength
    xi_z = k * (jnp.sqrt(1 - (xi_x + sx) ** 2 - (xi_y + sy) ** 2) - sz)
    factor = xiz_function(thickness / 2 * xi_z)
    factor = jnp.where(jnp.isnan(factor), 0, factor)

    f_coefficient = fft(interaction.reflection_coefficient())
    new_f_coefficient = f_coefficient * factor
    new_coefficient = ifft(new_f_coefficient)
    new_interaction = eqx.tree_at(lambda i: i.reflection_coefficient._data, interaction, new_coefficient)
    new_model = eqx.tree_at(lambda m: m.interaction, model, new_interaction)
    return new_model


@gin.configurable
def remove_detector_darkcounts(
    model: PtychographyModel,
    **kwargs,
) -> PtychographyModel:
    """Zero out the detector dark-count (background) correction.

    Replaces the detector's ``dark_counts`` array with zeros, effectively
    disabling dark-frame subtraction. Useful when reusing a model that was
    initialized with measured dark counts but the current reconstruction
    should not apply that correction.

    Args:
        model: The ptychography model to modify.

    Returns:
        A new :py:class:`PtychographyModel` with zeroed dark counts.
    """
    new_detector = eqx.tree_at(
        lambda d: d.dark_counts,
        model.detector,
        jnp.zeros_like(model.detector.dark_counts),
    )
    new_model = eqx.tree_at(lambda m: m.detector, model, new_detector)
    return new_model


@gin.configurable
def shift_probe_and_interaction(
    model: PtychographyModel,
    epoch: int,
    optimizer_state: optax.OptState,
    apply_every: int = 1,
    max_epoch: int = 1800,
    compute_shift_fn: Callable[[Inexact[Array, "m n"]], Float[Array, "2"]] = compute_center_of_mass_shift,
    perform_shift_fn: Callable[[Inexact[Array, "m n"]], Inexact[Array, "... m n"]] = shift_image,
    probe_re: str = ".*probe.*data",
    interaction_re: str = ".*interaction.*data",
    order: int = 1,
    **kwargs,
) -> tuple[PtychographyModel, PyTree[PtychographyModel]]:
    """Center the probe by shifting both probe and interaction fields.

    Computes the center-of-mass offset of the probe amplitude and applies
    the corresponding sub-pixel shift to both the probe and interaction
    arrays (identified by regex on their pytree paths). The optimizer state
    arrays matching those paths are shifted as well to maintain consistency.

    This adjuster helps prevent local minima where the probe starts off-center
    and grows to hit the edge of the reconstruction grid.

    Args:
        model: The ptychography model to modify.
        epoch: Current epoch (used with ``apply_every`` and ``max_epoch``).
        optimizer_state: Current optimizer state; matching leaves are also shifted.
        apply_every: Only apply every ``apply_every`` epochs.
        max_epoch: Stop applying after this epoch.
        compute_shift_fn: Function to compute the 2-D shift from a spatial array.
        perform_shift_fn: Function to apply the shift to a spatial array.
        probe_re: Regex pattern matching probe data paths in the pytree.
        interaction_re: Regex pattern matching interaction data paths in the pytree.
        order: Interpolation order for the shift.

    Returns:
        A tuple ``(shifted_model, shifted_optimizer_state)``.
    """
    if epoch % apply_every != 0:
        return model, optimizer_state
    if epoch > max_epoch:
        return model, optimizer_state
    probe_img = jax.tree.leaves(
        jax.tree.map_with_path(lambda path, x: x if re.match(probe_re, make_path_string(path)) else None, model),
    )[0]
    compute_shift_fn = functools.partial(compute_shift_fn, order=order)
    any_re = f"({probe_re}|{interaction_re})"
    any_re_adam = f"({probe_re}|{interaction_re})&(.*mu.*|.*nu.*)"

    def non_spatial_axes(rank: int) -> tuple[int, ...]:
        return None if rank < 3 else tuple(range(rank - 3)) + (-1,)

    def nested_compute_shift(rank: int) -> Callable[[Inexact[Array, "... m n"]], Float[Array, "... 2"]]:
        axes_to_map = non_spatial_axes(rank)
        if axes_to_map is None:
            return compute_shift_fn
        return vmap(nested_compute_shift(rank - 1), in_axes=axes_to_map[0], out_axes=axes_to_map[0])

    shift = nested_compute_shift(len(probe_img.shape))(jnp.abs(probe_img))

    def nested_shift(rank: int) -> Callable[[Inexact[Array, "... m n"]], Float[Array, "... m n"]]:
        axes_to_map = non_spatial_axes(rank)
        if axes_to_map is None:
            return perform_shift_fn
        return vmap(nested_shift(rank - 1), in_axes=(axes_to_map[0], axes_to_map[0]), out_axes=axes_to_map[0])

    logging.info(f"Shifting by {shift}")

    new_model = jax.tree.map_with_path(
        lambda path, x: nested_shift(len(x.shape))(x, shift) if re.match(any_re, make_path_string(path)) else x, model
    )
    new_optimizer = jax.tree.map_with_path(
        lambda path, x: (
            nested_shift(len(x.shape))(x, shift) if re.match(any_re_adam, make_path_string(path)) and x.shape else x
        ),
        optimizer_state,
    )

    return new_model, new_optimizer


@gin.configurable
def dropout(
    model: PtychographyModel,
    epoch: int,
    optimizer_state: optax.OptState,
    apply_every: int = 1,
    max_epoch: int = 1800,
    probe_re: str = ".*probe.*data",
    interaction_re: str = ".*interaction.*data",
    fraction: float = 0.5,
    fraction_decay: float = 0.983,
    *,
    key: Optional[Key] = None,
) -> tuple[PtychographyModel, PyTree[PtychographyModel]]:
    """Apply random dropout to probe and interaction arrays.

    Randomly zeroes a fraction of elements in the probe and interaction
    data arrays (identified by regex on their pytree paths). The dropout
    fraction decays exponentially with epoch as
    ``fraction * fraction_decay ** epoch``. This regularizer can help
    prevent overfitting.

    Args:
        model: The ptychography model to modify.
        epoch: Current epoch (used with ``apply_every``, ``max_epoch``, and decay).
        optimizer_state: Current optimizer state (returned unchanged).
        apply_every: Only apply every ``apply_every`` epochs.
        max_epoch: Stop applying after this epoch.
        probe_re: Regex pattern matching probe data paths in the pytree.
        interaction_re: Regex pattern matching interaction data paths.
        fraction: Base dropout fraction (probability of zeroing each element).
        fraction_decay: Exponential decay rate for the dropout fraction per epoch.
        key: JAX PRNG key for generating the random dropout mask.

    Returns:
        A tuple ``(model_with_dropout, optimizer_state)``.

    Raises:
        ValueError: If ``key`` is ``None``.
    """
    if epoch % apply_every != 0:
        return model, optimizer_state
    if epoch > max_epoch:
        return model, optimizer_state
    any_re = f"({probe_re}|{interaction_re})"

    def dropout_fn(x: Inexact[Array, "... m n"]) -> Inexact[Array, "... m n"]:
        if key is None:
            raise ValueError("Key must be provided for dropout")
        dropout_mask = jax.random.bernoulli(key, p=1 - fraction * fraction_decay**epoch, shape=x.shape)
        return x * dropout_mask

    new_model = jax.tree.map_with_path(
        lambda path, x: dropout_fn(x) if re.match(any_re, make_path_string(path)) else x, model
    )
    return new_model


@gin.configurable
def make_multislice(
    interaction: InteractionModel,
    interaction_generators: list[Callable[[InteractionModel], InteractionModel]],
    slice_displacements: list[float],
    separable_in_z: bool = False,
    inverted_bottom: bool = True,
    symmetric: bool = False,
) -> MultiSlice:
    """Convert a single-slice interaction model into a multi-slice model.

    Creates a :py:class:`~ptyrax.models.interaction.MultiSlice` by generating
    individual interaction slices from the given generators and stacking them
    at the specified z-displacements.

    Args:
        interaction: The base interaction model used as template for each slice.
        interaction_generators: List of callables that each produce a new
            interaction model from the base. If a single generator is provided,
            it is reused for all slices.
        slice_displacements: Z-positions (depths) of each slice. Length must
            match ``interaction_generators`` (or 1 generator is broadcast).
        separable_in_z: Whether to treat slices as separable in z during
            propagation.
        inverted_bottom: Whether the bottom slice uses an inverted coordinate
            frame (reflection geometry).
        symmetric: If ``True``, centers the slice displacements around zero.

    Returns:
        A :py:class:`~ptyrax.models.interaction.MultiSlice` model.

    Raises:
        ValueError: If the number of generators does not match the number of
            slice displacements.
    """
    if len(interaction_generators) == 1:
        interaction_generators *= len(slice_displacements)
    if len(interaction_generators) != len(slice_displacements):
        raise ValueError(
            f"Number of interaction generators {len(interaction_generators)} does not match "
            f"number of slice displacements {len(slice_displacements)}"
        )
    if symmetric:
        slice_displacements = [d - np.mean(slice_displacements) for d in slice_displacements]
    new_inner_interactions = [gen(interaction) for gen in interaction_generators]
    return MultiSlice.from_interactions(
        new_inner_interactions, slice_displacements, separable_in_z=separable_in_z, inverted_bottom=inverted_bottom
    )


@gin.configurable
def offset_displacement(
    model: PtychographyModel,
    offset: tuple[float, float],
) -> PtychographyModel:
    """Add a constant offset to the multi-slice inter-slice distances.

    Shifts all ``slice_distances`` in the interaction model by a fixed offset.
    This is useful for adjusting the nominal depth separation between slices
    in a multi-slice reconstruction.

    Args:
        model: The ptychography model to modify (must have a multi-slice interaction).
        offset: Offset to add to each slice distance, as a tuple ``(dz_0, dz_1, ...)``.

    Returns:
        A new :py:class:`PtychographyModel` with adjusted slice distances.
    """
    interaction = model.interaction
    new_displacement = interaction.slice_distances + jnp.array(offset)
    new_interaction = eqx.tree_at(lambda i: i.slice_distances, interaction, new_displacement)
    return eqx.tree_at(lambda m: m.interaction, model, new_interaction)


@gin.configurable
def set_mean_phase(interaction: InteractionModel, target_mean_phase: float = 0.0) -> InteractionModel:
    """Shift the global phase of the interaction reflection coefficient.

    Applies a constant phase rotation to the reflection coefficient so that
    its spatial mean matches ``target_mean_phase``. This removes global phase
    ambiguity that can accumulate during optimization.

    Args:
        interaction: The interaction model to modify.
        target_mean_phase: Desired mean phase angle (radians) of the reflection
            coefficient.

    Returns:
        A new :py:class:`~ptyrax.models.interaction.InteractionModel` with the
        adjusted phase.
    """
    interaction_data = interaction.reflection_coefficient()
    current_mean_phase = jnp.angle(jnp.mean(interaction_data))
    phase_shift = target_mean_phase - current_mean_phase
    new_interaction_data = interaction_data * phase_only_exp(phase_shift)
    new_interaction = eqx.tree_at(
        lambda i: i.reflection_coefficient,
        interaction,
        type(interaction.reflection_coefficient)(new_interaction_data),
    )
    return new_interaction


@gin.configurable
def reinitialize_interaction(
    interaction: InteractionModel,
    new_data_initializer: Callable[[tuple[int, int]], InteractionModel],
    new_type: Type[InteractionModel] = None,
) -> InteractionModel:
    """Reinitialize the interaction model with a new data initializer.

    Creates a fresh interaction model of the same (or different) type using
    the existing coordinate system and sampling grids but with newly
    generated reflection coefficient data.

    Args:
        interaction: The current interaction model (used for coordinates,
            sampling, and regularization functions).
        new_data_initializer: Callable that generates new coefficient data
            given a :py:class:`~ptyrax.spatial.SamplingGrid`.
        new_type: Optional alternative interaction class. If ``None``, uses
            the same type as the input.

    Returns:
        A new :py:class:`~ptyrax.models.interaction.InteractionModel` with
        reinitialized data.
    """
    if new_type is None:
        new_type = type(interaction)

    new_interaction = new_type(
        coordinates=interaction.coordinates,
        sampling=interaction.sampling,
        initializer=new_data_initializer,
        regularization_functions=interaction.regularization_functions,
    )
    return new_interaction


# TODO remove this function, it is for a very specific use case.
@gin.configurable
def reinitialize_interaction_inverted(
    interaction: InteractionModel,
    new_data_initializer: Callable[[tuple[int, int]], InteractionModel],
    new_type: Type[InteractionModel] = None,
) -> InteractionModel:
    """Reinitialize the interaction model with inverted amplitude.

    Like :py:func:`reinitialize_interaction`, but inverts the amplitude of the
    generated data: pixels with high amplitude in the initializer get low
    amplitude in the result and vice versa. Phase is set to zero. This is
    useful for initializing complementary or "negative" interaction patterns.

    Args:
        interaction: The current interaction model (used for coordinates,
            sampling, and regularization functions).
        new_data_initializer: Callable that generates coefficient data
            given a :py:class:`~ptyrax.spatial.SamplingGrid`.
        new_type: Optional alternative interaction class. If ``None``, uses
            the same type as the input.

    Returns:
        A new :py:class:`~ptyrax.models.interaction.InteractionModel` with
        amplitude-inverted data.
    """
    if new_type is None:
        new_type = type(interaction)

    def inverted_initializer(*args, **kwargs) -> Array:
        data = new_data_initializer(*args, **kwargs)
        amp = jnp.abs(data)
        new_amp = (1 - amp / jnp.max(amp)) * jnp.max(amp)
        new_data = new_amp * jnp.exp(1j * 0 * jnp.angle(data))  # 0: Constant phase
        return new_data

    new_interaction = new_type(
        coordinates=interaction.coordinates,
        sampling=interaction.sampling,
        initializer=inverted_initializer,
        regularization_functions=interaction.regularization_functions,
    )
    return new_interaction


@gin.configurable
def reset_interaction(
    model: PtychographyModel,
    initial_model: PtychographyModel,
) -> PtychographyModel:
    """Reset the interaction model to its initial state.

    Replaces the current (optimized) interaction model with the one from
    ``initial_model``. This is useful for restarting the interaction
    optimization from scratch while preserving other model components.

    Args:
        model: The current ptychography model.
        initial_model: The model whose interaction will be copied.

    Returns:
        A new :py:class:`PtychographyModel` with the initial interaction.
    """
    new_model = eqx.tree_at(lambda m: m.interaction, model, initial_model.interaction)
    return new_model


@gin.configurable
def replace_illumination(
    model: PtychographyModel,
    new_illumination_generator: Callable[[IlluminationModel], IlluminationModel] = identity,
) -> PtychographyModel:
    """Replace the model's illumination with a transformed version.

    Applies a generator function to the current illumination model and
    substitutes the result into the ptychography model.

    Args:
        model: The ptychography model to modify.
        new_illumination_generator: Callable taking the current
            :py:class:`~ptyrax.models.illumination.IlluminationModel` and
            returning a new one.

    Returns:
        A new :py:class:`PtychographyModel` with the replaced illumination.
    """
    new_illumination = new_illumination_generator(model.illumination)

    new_model = eqx.tree_at(lambda m: m.illumination, model, new_illumination)
    return new_model


@gin.configurable
def normalize_illumination(
    model: PtychographyModel,
    **kwargs,
) -> PtychographyModel:
    r"""Normalize the probe illumination to unit total energy.

    Divides the probe data by its L2 norm so that
    :math:`\sum |\mathrm{probe}|^2 = 1`. This removes amplitude ambiguity
    between probe and interaction during optimization.

    Args:
        model: The ptychography model to modify.

    Returns:
        A new :py:class:`PtychographyModel` with the normalized probe.
    """
    probe_data = model.illumination.probe()
    norm = jnp.sqrt(jnp.sum(jnp.abs(probe_data) ** 2))
    new_probe_data = probe_data / norm
    new_probe_data = (
        type(model.illumination.probe.data)(new_probe_data)
        if isinstance(model.illumination.probe.data, ArrayParametrization)
        else new_probe_data
    )
    new_illumination = eqx.tree_at(lambda i: i.probe.data, model.illumination, new_probe_data)
    new_model = eqx.tree_at(lambda m: m.illumination, model, new_illumination)
    return new_model


@gin.configurable
def replace_illumination_from_hdf5(
    model: PtychographyModel,
    hdf5_path: str,
    hdf5_illumination_path: str = "illumination",
    illumination_adjustment_fns: list[Callable[[IlluminationModel], IlluminationModel]] = None,
    data_only: bool = False,
    normalize: bool = False,
    **kwargs,
) -> PtychographyModel:
    """Replace the model's illumination with parameters loaded from an HDF5
    file.

    Loads previously saved illumination state from an HDF5 reconstruction file.
    Optionally applies a sequence of adjustment functions after loading, and can
    restrict the replacement to data-only (interpolated to the current grid) with
    optional normalization.

    Args:
        model: The ptychography model to modify.
        hdf5_path: Path to the HDF5 file containing saved illumination parameters.
            If ``None`` or empty, the function returns the model unchanged.
        hdf5_illumination_path: Group path prefix within the HDF5 file.
        illumination_adjustment_fns: Optional list of callables applied
            sequentially to the loaded illumination.
        data_only: If ``True``, only the probe data array is replaced
            (interpolated to match the current model's grid), leaving
            coordinates and metadata unchanged.
        normalize: If ``True`` and ``data_only=True``, normalize the
            loaded probe data to unit energy.
        **kwargs: Additional keyword arguments forwarded to
            :py:func:`~ptyrax.hdf5_checkpoint.apply_hdf5_to_model`.

    Returns:
        A new :py:class:`PtychographyModel` with illumination loaded from HDF5.
    """
    if illumination_adjustment_fns is None:
        illumination_adjustment_fns = []
    if hdf5_path is None or not hdf5_path:
        logging.warning("No HDF5 path provided, skipping illumination replacement.")
        return model
    new_illumination, _, _ = apply_hdf5_to_model(
        model.illumination,
        hdf5_path,
        path_prefix=hdf5_illumination_path,
        **kwargs,
    )
    for fn in illumination_adjustment_fns:
        new_illumination = fn(new_illumination, initial_illumination=model.illumination)

    if data_only:
        new_data = new_illumination._probe()[0, ..., 0]
        # new_data = resize_to_match(new_data[0, :, :, 0], model.illumination._probe.sampling.shape)[
        #     jnp.newaxis, :, :, jnp.newaxis
        # ]
        new_data = interpolate_grid_to_grid(
            new_data,
            source_grid=new_illumination.probe.sampling[0],
            target_grid=model.illumination.probe.sampling[0],
            interpolation_mode="amplitude_phase",
        )[jnp.newaxis, :, :, jnp.newaxis]
        new_data = new_data / jnp.sqrt(jnp.sum(jnp.abs(new_data) ** 2)) if normalize else new_data
        new_illumination = eqx.tree_at(
            lambda i: i.probe.data,
            model.illumination,
            type(model.illumination.probe.data)(new_data)
            if isinstance(model.illumination.probe.data, ArrayParametrization)
            else new_data,
        )

    new_model = eqx.tree_at(lambda m: m.illumination, model, new_illumination)
    return new_model


@gin.configurable
def add_wavelength_channels(
    model: PtychographyModel,
    initial_model: PtychographyModel,
    additional_wavelengths: int = 0,
) -> PtychographyModel:
    """Extend the illumination model with additional wavelength channels.

    Pads the leading (wavelength) dimension of the probe data array with
    ``additional_wavelengths`` new entries initialized to the defaults from
    ``initial_model``. This enables polychromatic reconstructions by growing
    the spectral dimension during training.

    Args:
        model: The current ptychography model.
        initial_model: Model providing default values for the new channels.
        additional_wavelengths: Number of wavelength channels to add.

    Returns:
        A new :py:class:`PtychographyModel` with extended wavelength dimension.
    """

    def pad_pytree(
        pytree: PyTree[Shaped, " T"],  # pyright: ignore[reportUndefinedVariable]
        n: int,
        defaults: PyTree[Shaped, " T"],  # pyright: ignore[reportUndefinedVariable]
    ) -> PyTree[Shaped, " T"]:  # pyright: ignore[reportUndefinedVariable]
        """Extend the leading dimension of each array in pytree by N, filling
        new entries with defaults (same pytree structure)."""
        return jax.tree.map(
            lambda arr, d: jnp.concatenate([arr, jnp.full((n,) + arr.shape[1:], d, arr.dtype)]), pytree, defaults
        )

    padded_data = pad_pytree(model.illumination.probe.data, additional_wavelengths)
    new_model = eqx.tree_at(lambda m: m.illumination.probe.data, model, padded_data)
    return new_model


@gin.configurable
def reset_illumination(
    model: PtychographyModel,
    initial_model: PtychographyModel,
) -> PtychographyModel:
    """Reset the illumination model to its initial state.

    Replaces the current (optimized) illumination with the one from
    ``initial_model``. This is useful for restarting the illumination
    optimization from scratch while preserving other model components.

    Args:
        model: The current ptychography model.
        initial_model: The model whose illumination will be copied.

    Returns:
        A new :py:class:`PtychographyModel` with the initial illumination.
    """
    new_model = eqx.tree_at(lambda m: m.illumination, model, initial_model.illumination)
    return new_model


@gin.configurable()
def limit_illumination_na(
    model: PtychographyModel,
    max_na: float | tuple[float, float],
) -> DirectIllumination:
    """Apply a numerical aperture limit to the illumination probe.

    Propagates the probe to Fourier (angular) space, applies an elliptical
    aperture mask defined by ``max_na``, and propagates back. Frequencies
    beyond the NA limit are zeroed, effectively band-limiting the probe.

    Args:
        model: The ptychography model to modify. Must use
            :py:class:`~ptyrax.models.illumination.DirectIllumination`.
        max_na: Maximum numerical aperture as a scalar (isotropic) or a tuple
            ``(na_x, na_y)`` for anisotropic limiting.

    Returns:
        A new :py:class:`PtychographyModel` with the NA-limited probe.

    Raises:
        ValueError: If ``max_na`` is not a scalar or 2-element tuple, or if
            the illumination is not ``DirectIllumination``.
    """
    max_na = np.array(max_na)
    if max_na.ndim == 0:
        max_na = np.array([max_na, max_na])
    if max_na.shape != (2,):
        raise ValueError(f"max_na must be a scalar or a tuple of two floats. got: {max_na}")
    illumination = model.illumination
    if not isinstance(illumination, DirectIllumination):
        raise ValueError("NA limiting is only implemented for DirectIllumination models.")

    def limit_per_mode(illumination_mode: CoherentField) -> CoherentField:
        pupil = illumination_mode.propagate_fraunhofer(1.0)  # 1.0: equivalent to angular space
        kx, ky = pupil.sampling.meshgrid
        kr = (kx / max_na[0]) ** 2 + (ky / max_na[1]) ** 2
        aperture_mask = kr <= 1.0
        limited_pupil_data = pupil() * aperture_mask[..., jnp.newaxis]
        limited_pupil = eqx.tree_at(
            lambda p: p.data,
            pupil,
            limited_pupil_data,
        )
        limited_field = limited_pupil.propagate_fraunhofer(1.0, inverse=True)
        return limited_field

    na_limited_fields = eqx.filter_vmap(limit_per_mode)(illumination())
    na_limited_illumination = eqx.tree_at(
        lambda p: p._probe,
        illumination,
        na_limited_fields,
    )

    model = eqx.tree_at(lambda m: m.illumination, model, na_limited_illumination)

    return model


@gin.configurable
def set_illumination_total_energy(
    model: PtychographyModel,
    total_energy: float = 1.0,
) -> PtychographyModel:
    r"""Scale the probe so that its total integrated intensity equals a target
    value.

    Computes the current total energy (sum of probe intensities) and scales
    the probe data by $\sqrt{E_{\text{target}} / E_{\text{current}}}$ to achieve
    the desired total energy.

    Args:
        model: The ptychography model to modify.
        total_energy: Desired total probe energy (sum of pixel intensities).

    Returns:
        A new :py:class:`PtychographyModel` with the scaled probe.
    """
    probe = model.illumination(0)
    current_energy = jnp.sum(probe())
    scale_factor = total_energy / current_energy
    new_probe_data = model.illumination.probe.data * jnp.sqrt(scale_factor)
    new_probe_data = (
        type(model.illumination.probe.data)(new_probe_data)
        if isinstance(model.illumination.probe.data, ArrayParametrization)
        else new_probe_data
    )
    new_model = eqx.tree_at(lambda m: m.illumination.probe.data, model, new_probe_data)
    return new_model


@gin.configurable
def scale_scan_range(
    model: PtychographyModel,
    scale: float = 1.0,
) -> PtychographyModel:
    """Scale the sample scan positions by a multiplicative factor.

    Multiplies all interaction coordinate translations by ``scale``.
    This is useful for correcting miscalibrated scan step sizes or
    converting between units.

    Args:
        model: The ptychography model to modify.
        scale: Multiplicative factor for all scan translations.

    Returns:
        A new :py:class:`PtychographyModel` with rescaled scan positions.
    """
    resolved_model = resolve_parametrizations(model)
    translations = resolved_model.interaction.coordinates._translation
    new_translations = translations * scale
    new_model = eqx.tree_at(
        lambda m: m.interaction.coordinates.parametrizations._translation,
        model,
        new_translations,
    )
    return new_model


@gin.configurable
def limit_reflection_NA(
    model: PtychographyModel,
    max_na: float | tuple[float, float],
) -> InteractionModel:
    """Apply a numerical aperture limit to the interaction reflection
    coefficient.

    Multiplies the reflection coefficient by an elliptical aperture mask in
    real space (which corresponds to a Fourier-space NA limit for the reflected
    field). Frequencies beyond ``max_na`` are zeroed.

    Args:
        model: The ptychography model to modify. Must use
            :py:class:`~ptyrax.models.interaction.FresnelReflection`.
        max_na: Maximum numerical aperture as a scalar (isotropic) or a tuple
            ``(na_x, na_y)`` for anisotropic limiting.

    Returns:
        A new :py:class:`PtychographyModel` with the NA-limited interaction.

    Raises:
        ValueError: If ``max_na`` is not a scalar or 2-element tuple, or if
            the interaction is not ``FresnelReflection``.
    """
    max_na = np.array(max_na)
    if max_na.ndim == 0:
        max_na = np.array([max_na, max_na])
    if max_na.shape != (2,):
        raise ValueError(f"max_na must be a scalar or a tuple of two floats. got: {max_na}")
    interaction = model.interaction
    if not isinstance(interaction, FresnelReflection):
        raise ValueError("NA limiting is only implemented for FresnelReflection models.")

    kx, ky = interaction.sampling.meshgrid
    kr = (kx / max_na[0]) ** 2 + (ky / max_na[1]) ** 2
    aperture_mask = kr <= 1.0
    limited_reflection_data = interaction.reflection_coefficient() * aperture_mask[..., jnp.newaxis]
    limited_interaction = eqx.tree_at(
        lambda i: i.reflection_coefficient,
        interaction,
        limited_reflection_data,
    )

    model = eqx.tree_at(lambda m: m.interaction, model, limited_interaction)

    return model


# endregion
