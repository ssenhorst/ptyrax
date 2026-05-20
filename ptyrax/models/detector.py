from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Dict, Literal, Optional, Union

import equinox as eqx
import gin
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as JaxRotation
from jaxtyping import Array, Float, PyTree
from tensorboardX import SummaryWriter

from ptyrax.field import CoherentField
from ptyrax.logger import log_image
from ptyrax.models.illumination import NamedLoss
from ptyrax.models.propagation import (
    source_fourier_occupancy_from_tilt_interpolation,
    source_fourier_support_from_tilt_interpolation,
)
from ptyrax.parametrizations import IndexSliceParameter, resolve_array_parametrizations
from ptyrax.spatial import CoordinateSystem, Rotation, SamplingGrid
from ptyrax.state_geometry import state_get_with_candidates


@gin.configurable()
class Detector(eqx.Module):
    """Abstract detector interface mapping input fields to measured detector
    counts."""

    coordinates: IndexSliceParameter[CoordinateSystem]
    sampling: SamplingGrid = eqx.field(static=True)

    @abstractmethod
    def __call__(self, input_field: CoherentField, index: Optional[int]) -> Float[Array, "* m n"]:
        pass

    @classmethod
    def load_from_hdf5_state(
        cls,
        state: Dict[str, np.ndarray],
        *,
        detector_path_prefix: str = "detector",
    ) -> "Detector":
        r"""Construct a concrete detector instance from an HDF5 state
        dictionary.

        Extracts rotation, translation, pixel size, and optionally dark counts
        from the HDF5 state and builds the detector coordinate system and
        sampling grid.

        Args:
            state: Dictionary mapping HDF5 dataset paths to NumPy arrays,
                as returned by reading an HDF5 checkpoint file.
            detector_path_prefix: Path prefix within the state dictionary
                under which detector parameters are stored.

        Returns:
            An instance of the concrete detector subclass populated from the
            state dictionary.

        Raises:
            TypeError: If called directly on the abstract
                :py:class:`~ptyrax.models.detector.Detector` base class.
            KeyError: If the required ``dark_counts`` key is missing from
                the state dictionary.
            ValueError: If the detector pixel size has fewer than one value.

        Example:
            >>> state = load_hdf5_to_dict("reconstruction.hdf5")
            >>> detector = NoiselessEqualWeightDetector.load_from_hdf5_state(state, detector_path_prefix="detector")
        """
        if cls is Detector:
            raise TypeError("Call `load_from_hdf5_state` on a concrete Detector subclass.")

        prefix = detector_path_prefix.strip("/")
        pref = f"{prefix}/" if prefix else ""

        rotation_6d = np.asarray(
            state_get_with_candidates(
                state,
                [
                    f"{pref}coordinates/parameters/rotation/_representation_6d",
                    f"{pref}coordinates/rotation/_representation_6d",
                ],
            )
        )
        translation = np.asarray(
            state_get_with_candidates(
                state,
                [
                    f"{pref}coordinates/parameters/_translation",
                    f"{pref}coordinates/_translation",
                    f"{pref}coordinates/translation",
                ],
            )
        )
        detector_pixel_size = np.asarray(
            state_get_with_candidates(
                state,
                [f"{pref}sampling/pixel_size"],
            )
        ).reshape(-1)

        if detector_pixel_size.size == 1:
            detector_pixel_size = np.repeat(detector_pixel_size, 2)
        if detector_pixel_size.size < 2:
            raise ValueError("Detector pixel size must have at least one value.")

        if rotation_6d.ndim == 1:
            rotation_6d = rotation_6d[None, :]
        if translation.ndim == 1:
            translation = translation[None, :]

        if f"{pref}dark_counts" in state:
            dark_counts = np.asarray(state[f"{pref}dark_counts"])
            detector_shape = tuple(dark_counts.shape[-2:])
        else:
            dark_counts = None
            raise KeyError(f"Missing required detector shape source '{pref}dark_counts' in state.")

        coordinates = IndexSliceParameter(
            CoordinateSystem(
                rotation=Rotation(jnp.asarray(rotation_6d)),
                translation=jnp.asarray(translation),
            )
        )
        sampling = SamplingGrid.from_tuples(detector_shape, tuple(detector_pixel_size[:2]))

        kwargs = {
            "coordinates": coordinates,
            "sampling": sampling,
        }
        if dark_counts is not None and "dark_counts" in getattr(cls, "__annotations__", {}):
            kwargs["dark_counts"] = jnp.asarray(dark_counts)

        try:
            return cls(**kwargs)
        except TypeError:
            kwargs.pop("dark_counts", None)
            return cls(**kwargs)

    @classmethod
    def from_hdf5_state(
        cls,
        state: Dict[str, np.ndarray],
        *,
        detector_path_prefix: str = "detector",
    ) -> "Detector":
        """Alias for :py:meth:`load_from_hdf5_state`.

        Args:
            state: Dictionary mapping HDF5 dataset paths to NumPy arrays.
            detector_path_prefix: Path prefix for detector parameters in
                the state dictionary.

        Returns:
            An instance of the concrete detector subclass.
        """
        return cls.load_from_hdf5_state(state, detector_path_prefix=detector_path_prefix)

    def _resolve_target_coordinate_system(self, index: int = 0):
        """Resolve the target coordinate system for a given index."""
        coords = resolve_array_parametrizations(self.coordinates)
        if hasattr(coords, "at"):
            return coords.at(index)
        elif hasattr(coords, "at_current_index"):
            return coords.at_current_index()
        return coords

    def fourier_mask(
        self,
        field: CoherentField,
        *,
        index: int = 0,
        return_details: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute the Fourier-space occupancy mask for this detector.

        Determines which pixels in the source field's Fourier representation
        map onto the detector, given the relative tilt between the field
        coordinate system and the detector coordinate system.

        Args:
            field: The coherent source field whose Fourier occupancy is
                evaluated against the detector geometry.
            index: Position index into the detector coordinate parameter,
                used when the detector has multiple indexed orientations.
            return_details: If ``True``, return the full tuple of
                intermediate arrays instead of just the source occupancy.

        Returns:
            If ``return_details`` is ``False``, a 2-D array of the source
            Fourier occupancy (fraction of each source pixel captured by
            the detector). If ``True``, a tuple of
            ``(source_occupancy, detector_mask, field_frame_target_indices)``.
        """
        target_coordinate_system = self._resolve_target_coordinate_system(index)

        source_occupancy, detector_mask, field_frame_target_indices = source_fourier_occupancy_from_tilt_interpolation(
            field,
            target_coordinate_system,
            self.sampling,
        )
        if return_details:
            return source_occupancy, detector_mask, field_frame_target_indices
        return source_occupancy

    def fourier_support_mask(
        self,
        field: CoherentField,
        *,
        index: int = 0,
        return_details: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute the Fourier-space binary support mask for this detector.

        Similar to :py:meth:`fourier_mask`, but returns a binary support
        indicating which source Fourier pixels have *any* overlap with the
        detector, rather than a fractional occupancy.

        Args:
            field: The coherent source field whose Fourier support is
                evaluated against the detector geometry.
            index: Position index into the detector coordinate parameter.
            return_details: If ``True``, return the full tuple of
                intermediate arrays.

        Returns:
            If ``return_details`` is ``False``, a 2-D binary array of the
            source Fourier support. If ``True``, a tuple of
            ``(source_support, source_occupancy, detector_mask,
            field_frame_target_indices)``.
        """
        target_coordinate_system = self._resolve_target_coordinate_system(index)

        source_support, source_occupancy, detector_mask, field_frame_target_indices = (
            source_fourier_support_from_tilt_interpolation(
                field,
                target_coordinate_system,
                self.sampling,
            )
        )
        if return_details:
            return source_support, source_occupancy, detector_mask, field_frame_target_indices
        return source_support

    def __log_epoch__(
        self,
        writer: SummaryWriter,
        epoch: int,
        prefix: str = "",
        **kwargs,
    ) -> None:
        """Log detector pose parameters to TensorBoard.

        Writes the detector Euler angles (xyz convention, in degrees) and
        translation components as scalar summaries under the
        ``3_detector/`` tag group.

        Args:
            writer: TensorBoard summary writer instance.
            epoch: Current training epoch number.
            prefix: Optional prefix for TensorBoard tags (unused in this
                base implementation but accepted for interface consistency).
            **kwargs: Additional keyword arguments (ignored).
        """
        coordinate = resolve_array_parametrizations(self.coordinates).at(0)
        position = coordinate.translation
        r = np.array(coordinate.rotation.as_matrix())
        euler_angles = JaxRotation.from_matrix(r).as_euler("xyz", degrees=True)
        writer.add_scalar("3_detector/rotation/x", euler_angles[0], epoch)
        writer.add_scalar("3_detector/rotation/y", euler_angles[1], epoch)
        writer.add_scalar("3_detector/rotation/z", euler_angles[2], epoch)
        writer.add_scalar("3_detector/position/x", position[0], epoch)
        writer.add_scalar("3_detector/position/y", position[1], epoch)
        writer.add_scalar("3_detector/position/z", position[2], epoch)
        writer.add_scalar(
            "3_detector/position/distance",
            np.linalg.norm(position),
            epoch,
        )


@gin.configurable()
class NoiselessEqualWeightDetector(Detector):
    r"""Ideal noiseless detector with equal weighting across coherent modes.

    Models a detector that sums the intensity contributions from all coherent
    modes with equal weight and no noise or background. The output can be
    returned as either amplitude (square root of summed intensity) or raw
    intensity, controlled by the ``mode`` parameter.

    The forward model is:

    .. math::

        I_{\text{det}} = \sum_k |\psi_k|^2

    where :math:`\psi_k` are the coherent mode fields at the detector plane.

    Attributes:
        coordinates: Detector coordinate system (position and orientation).
        sampling: Detector pixel grid specification.
        mode: Output mode — ``"amplitude"`` returns
            :math:`\sqrt{I_{\text{det}}}`, ``"intensity"`` returns
            :math:`I_{\text{det}}` directly.

    Example:
        >>> detector = NoiselessEqualWeightDetector(
        ...     coordinates=detector_coords,
        ...     sampling=detector_sampling,
        ...     mode="amplitude",
        ... )
        >>> predicted = detector(exit_fields)
    """

    coordinates: IndexSliceParameter[CoordinateSystem]
    sampling: SamplingGrid = eqx.field(static=True)
    mode: Union[Literal["amplitude"], Literal["intensity"]] = eqx.field(static=True, default="amplitude")

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the noiseless equal-weight detector.

        Args:
            *args: Positional arguments forwarded to
                :py:class:`~ptyrax.models.detector.Detector`.
            **kwargs: Keyword arguments forwarded to the parent class.
                The ``mode`` keyword is consumed here and not forwarded.
        """
        mode = kwargs.pop("mode", "amplitude")
        self.mode = mode
        super().__init__(*args, **kwargs)

    def __call__(self, input_fields: PyTree[CoherentField]) -> Float[Array, "* m n"]:
        r"""Compute the noiseless detector measurement from input fields.

        Sums intensity contributions from all coherent modes and returns
        either the amplitude or intensity depending on the detector mode.

        Args:
            input_fields: PyTree of coherent fields at the detector plane,
                with an ``.intensity`` attribute of shape
                ``(n_modes, m, n, 1)``.

        Returns:
            A 2-D array of shape ``(m, n)`` containing the predicted
            detector readout (amplitude or intensity).
        """
        coherent_counts = input_fields.intensity[..., 0]
        incoherent_counts = jnp.sum(coherent_counts, axis=0)
        return jnp.sqrt(incoherent_counts) if self.mode == "amplitude" else incoherent_counts


@gin.configurable()
class BackgroundEqualWeightDetector(Detector):
    r"""Detector model with learnable dark-count background.

    Extends the equal-weight detector by adding a trainable 2-D
    ``dark_counts`` array that models detector background (e.g. thermal
    noise, stray light). The forward model averages amplitudes across
    coherent modes and adds the dark counts:

    .. math::

        I_{\text{det}} = \frac{1}{K}\sum_k |\psi_k| + D

    where :math:`D` is the dark-count background and :math:`K` is the
    number of coherent modes.

    Regularization functions can be attached to penalize the dark-count
    map during optimization (e.g. smoothness or sparsity priors).

    Attributes:
        coordinates: Detector coordinate system.
        sampling: Detector pixel grid specification.
        dark_counts: Trainable 2-D background array of shape ``(m, n)``.
        dynamic_range: Tuple ``(min, max)`` of the detector dynamic range.
        scale: Multiplicative scale factor (static).
        regularization_functions: Tuple of callables applied to
            ``dark_counts`` to compute regularization losses.

    Example:
        >>> detector = BackgroundEqualWeightDetector(
        ...     coordinates=detector_coords,
        ...     sampling=detector_sampling,
        ...     dark_counts=jnp.zeros((256, 256)),
        ... )
        >>> predicted = detector(exit_fields)
    """

    coordinates: CoordinateSystem
    sampling: SamplingGrid
    dark_counts: Array
    dynamic_range: tuple[float, float] = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    regularization_functions: tuple[Callable[[Float[Array, "m n"]], float]] = eqx.field(static=True)

    def __init__(
        self,
        *args,
        dynamic_range: tuple[float, float] = None,
        dark_counts: Float[Array, "m n"] = None,
        scale: float = 1.0,
        regularization_functions: tuple[Callable[[Float[Array, "m n"]], float]] = (),
        **kwargs,
    ) -> None:
        """Initialize the background equal-weight detector.

        Args:
            *args: Positional arguments forwarded to
                :py:class:`~ptyrax.models.detector.Detector`.
            dynamic_range: Tuple of ``(min, max)`` detector counts. Defaults
                to ``(0.0, 2**16)`` if not provided.
            dark_counts: Initial 2-D dark-count background array. If
                ``None``, initialized to zeros matching the detector shape.
            scale: Static multiplicative scale factor applied to the
                detector model.
            regularization_functions: Tuple of callables that accept the
                dark-count array and return a scalar loss value.
            **kwargs: Additional keyword arguments forwarded to the parent
                class.
        """
        super(BackgroundEqualWeightDetector, self).__init__(*args, **kwargs)
        if dark_counts is not None:
            self.dark_counts = dark_counts
        else:
            self.dark_counts = jnp.zeros((self.sampling.n_x, self.sampling.n_y))

        self.dynamic_range = (0.0, 2**16) if dynamic_range is None else dynamic_range
        self.regularization_functions = regularization_functions
        self.scale = scale

    def __call__(self, input_fields: PyTree[CoherentField]) -> Float[Array, "* m n"]:
        """Compute the detector measurement including dark-count background.

        Averages the amplitude across coherent modes and adds the
        learned dark-count background.

        Args:
            input_fields: PyTree of coherent fields at the detector plane,
                with an ``.amplitude`` attribute of shape
                ``(n_modes, m, n, 1)``.

        Returns:
            A 2-D array of shape ``(m, n)`` containing the predicted
            detector readout (mean amplitude plus dark counts).
        """
        coherent_counts = input_fields.amplitude[..., 0]
        incoherent_counts = jnp.mean(coherent_counts, axis=0)
        total_detected_counts = incoherent_counts + self.dark_counts
        return total_detected_counts

    def __log_epoch__(
        self,
        writer: SummaryWriter,
        epoch: int,
        prefix: str = "",
        **kwargs,
    ) -> None:
        """Log detector pose and dark-count image to TensorBoard.

        Extends the base class logging by additionally writing the
        dark-count background as an image summary with gamma correction.

        Args:
            writer: TensorBoard summary writer instance.
            epoch: Current training epoch number.
            prefix: Optional prefix for TensorBoard tags.
            **kwargs: Additional keyword arguments (ignored).
        """
        super(BackgroundEqualWeightDetector, self).__log_epoch__(writer, epoch, prefix)
        log_image(writer, "3_detector/dark_counts", self.dark_counts, epoch, gamma=0.33)

    def __regularize__(self) -> list[NamedLoss]:
        """Compute regularization losses on the dark-count background.

        Applies each function in ``regularization_functions`` to the
        ``dark_counts`` array and accumulates a total scalar loss.

        Returns:
            A tuple of ``(total_loss, regularizations)`` where
            ``total_loss`` is the sum of all regularization terms and
            ``regularizations`` is a list of
            :py:class:`~ptyrax.models.illumination.NamedLoss` entries.
        """
        regularizations = []
        total = 0.0
        for fn in self.regularization_functions:
            value = fn(self.dark_counts)
            total += value
            regularizations.append(NamedLoss(f"dark_counts.{fn.__name__}", value))
        return total, regularizations
