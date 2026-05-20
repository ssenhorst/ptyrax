from __future__ import annotations

import functools
import logging
from abc import abstractmethod
from typing import Callable, Self, Type, Union

import equinox as eqx
import gin
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jaxtyping import Array, ArrayLike, Complex, Float, Key, PyTree
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from tensorboardX import SummaryWriter

from ptyrax.field import CoherentField
from ptyrax.initializers import aperture
from ptyrax.logger import log_image
from ptyrax.parametrizations import ArrayParametrization, IndexSliceParameter
from ptyrax.spatial import CoordinateSystem, SamplingGrid, meshgrid
from ptyrax.utils import phase_only_exp, plot


class NamedLoss(eqx.Module):
    """A named loss value produced during regularization.

    Pairs a human-readable tag with a scalar loss value so that individual
    regularization contributions can be tracked and logged separately.

    Attributes:
        tag: Identifier string for this loss contribution (e.g.
            ``"probe.smoothness.0"``).
        value: Scalar loss value.
    """

    tag: str = eqx.field(static=True)
    value: float


class IlluminationModel(eqx.Module):
    """Abstract base class for illumination models in ptychographic
    reconstruction.

    An illumination model describes the probe beam that illuminates the sample.
    Subclasses must implement :py:meth:`__call__` to produce the probe field and
    :py:meth:`__regularize__` to compute any regularization losses on the probe
    parameters.
    """

    @abstractmethod
    def __call__(self) -> CoherentField:
        """Compute and return the probe illumination field.

        Returns:
            The coherent probe field used to illuminate the sample.
        """
        pass

    @abstractmethod
    def __regularize__(self) -> tuple[float, list[NamedLoss]]:
        """Compute regularization losses for the illumination model parameters.

        Returns:
            A tuple of (total_loss, named_losses) where *total_loss* is the
            scalar sum of all regularization contributions and *named_losses*
            is a list of :py:class:`~ptyrax.models.illumination.NamedLoss`
            instances for per-term logging.
        """
        pass


@gin.configurable()
class DirectIllumination(IlluminationModel):
    """Direct illumination model storing an explicit probe field.

    This is the simplest illumination model: it holds a single
    :py:class:`~ptyrax.field.CoherentField` as the probe and returns it
    unchanged on every call.  Supports optional reparametrization of the
    underlying array and per-probe regularization functions.

    Attributes:
        _probe: The stored coherent probe field.
        regularization_functions: Tuple of callables applied to the probe
            during regularization.
    """

    _probe: CoherentField
    regularization_functions: tuple[Callable[["DirectIllumination"], float], ...] = eqx.field(static=True)

    def __init__(
        self,
        probe_data: Float[Array, "* m n d"] | ArrayParametrization,
        *args,
        n_scan: int = 1,
        key: Key = None,
        mode_data_or_initializer: Union[Callable[[IlluminationModel], ArrayLike], ArrayLike] = None,
        parametrization_type: Type[ArrayParametrization] = None,
        regularization_functions: tuple[Callable[[IlluminationModel], float], ...] = (),
        **kwargs,
    ) -> None:
        """Initialize a direct illumination model.

        Args:
            probe_data: Array or parametrization containing the probe field
                samples.  Remaining positional args are forwarded to
                :py:class:`~ptyrax.field.CoherentField`.
            n_scan: Number of scan positions (unused by this model but
                accepted for interface compatibility).
            key: JAX PRNG key (unused but accepted for compatibility).
            mode_data_or_initializer: Optional mode data or factory (unused
                by this class, reserved for subclass compatibility).
            parametrization_type: If provided, wraps *probe_data* in the
                given :py:class:`~ptyrax.parametrizations.ArrayParametrization`
                before constructing the field.
            regularization_functions: Tuple of callables, each mapping the
                probe field to a scalar regularization loss.
            **kwargs: Additional keyword arguments forwarded to
                :py:class:`~ptyrax.field.CoherentField`.
        """
        if parametrization_type is not None:
            probe_data = parametrization_type(probe_data)
        self._probe = CoherentField(probe_data, *args, **kwargs)
        self.regularization_functions = regularization_functions

    def __call__(self) -> CoherentField:
        """Return the stored probe field.

        Returns:
            The coherent probe field.
        """
        return self.probe

    def __regularize__(self) -> tuple[float, list[NamedLoss]]:
        """Compute regularization losses over the probe field.

        Applies each function in *regularization_functions* to the probe and
        aggregates the results.

        Returns:
            A tuple of (total_loss, named_losses) where *total_loss* is the
            scalar sum and *named_losses* contains per-function, per-mode
            :py:class:`~ptyrax.models.illumination.NamedLoss` entries.
        """
        regularizations = []
        total = 0.0
        for fn in self.regularization_functions:
            regularization_values = vmap(fn)(self.probe)
            total += jnp.sum(regularization_values)
            regularizations.extend(
                NamedLoss(f"probe.{fn.__name__}.{i}", reg_value) for i, reg_value in enumerate(regularization_values)
            )
        return total, regularizations

    @classmethod
    def from_coherent_field(cls, probe: CoherentField, **kwargs) -> Self:
        """Construct a DirectIllumination from an existing CoherentField.

        Args:
            probe: A fully constructed coherent field to use as the probe.
            **kwargs: Additional keyword arguments forwarded to the
                constructor (e.g. *regularization_functions*).

        Returns:
            A new :py:class:`~ptyrax.models.illumination.DirectIllumination`
            instance wrapping the given field.
        """
        return cls(
            probe.data,
            probe.wavelength,
            probe.sampling,
            probe.coordinate_system,
            probe.propagation_direction,
            probe.spatial_dims,
            probe.vector_dim,
            **kwargs,
        )

    @property
    def probe(self) -> CoherentField:
        """The coherent probe field stored by this illumination model."""
        return self._probe

    def __plot__(self, *args, **kwargs) -> None:
        """Plot the probe field.

        Delegates to :py:meth:`CoherentField.__plot__`.
        """
        self.probe.__plot__(*args, **kwargs)

    def __log_epoch__(self, *args, **kwargs) -> None:
        """Log the probe field for the current training epoch.

        Delegates to :py:meth:`CoherentField.__log_epoch__`.
        """
        self.probe.__log_epoch__(*args, **kwargs)


# TODO change / remove this implementation: it was really for a specific use case and is not sufficiently general.
@gin.configurable()
def default_mode_initializer(
    mode_index: int,
    sampling: SamplingGrid,
    max_index: int,
    wavelength: float,
    probe_grid: SamplingGrid,
    coordinates: CoordinateSystem,
    key: jax.random.PRNGKey,
) -> CoherentField:
    r"""Create a default probe mode for multi-mode illumination models.

    Generates an aperture-based probe mode whose radius scales linearly with
    *mode_index* and applies a random amplitude modulation combined with a
    quadratic phase ramp to break symmetry between modes.

    Args:
        mode_index: Zero-based index of the mode being initialized.
        sampling: Sampling grid describing the pixel spacing.
        max_index: Total number of modes (used to scale the aperture radius).
        wavelength: Probe wavelength in consistent units.
        probe_grid: Sampling grid of the output probe field.
        coordinates: Coordinate system for the output field.
        key: JAX PRNG key for random amplitude generation.

    Returns:
        A :py:class:`~ptyrax.field.CoherentField` representing the
        initialized probe mode.
    """
    shape = sampling.shape
    default_radius = [
        (mode_index + 1) * shape[0] / (5 * max_index),
        (mode_index + 1) * shape[1] / (5 * max_index),
    ]
    data = (
        aperture(
            shape,
            radius=jnp.array(default_radius),
            defocus=0,
        )
        * jax.random.uniform(key, shape=shape)
    )[..., jnp.newaxis]
    xx, yy = meshgrid(data.shape, 1 / np.array(data.shape))
    rr = jnp.sqrt(xx**2 + yy**2)
    data = data * phase_only_exp(rr**2 * mode_index * 1000)[..., np.newaxis]
    output_probe_mode = CoherentField(data, wavelength, probe_grid, coordinates)

    return output_probe_mode


@gin.configurable()
def default_weight_initializer(shape: tuple[int, ...]) -> Float[Array, "..."]:
    """Create default mode weights for an orthogonalized probe.

    Returns weights that decay as :math:`1 / (2k + 1)` for uniform
    initialization across modes.

    Args:
        shape: Shape of the weight array, typically ``(n_scan, n_modes)``.

    Returns:
        An array of mode weights with the given shape.
    """
    return 1 / (2 * jnp.ones(shape) + 1)


@gin.configurable()
class OrthogonalizedProbe(IlluminationModel):
    """Illumination model that represents a set of orthogonal probe modes with
    per-scan singular values.

    The `probe_modes` are combined with `singular_values` to produce the effective probe for each scan index.
    """

    probe_modes: PyTree[CoherentField]
    singular_values: Array

    n_modes: int = eqx.field(static=True, default=1)
    n_scan: int = eqx.field(static=True)
    relaxed_orthogonalization: float = eqx.field(static=True, default=0.0)
    regularization_functions: tuple[Callable[[CoherentField], float], ...] = eqx.field(static=True)

    # TODO refactor to fix the underscore, it's ugly
    # TODO also fix this mode data or initializer nonsense
    def __init__(
        self,
        _,
        wavelength: float,
        probe_grid: SamplingGrid,
        coordinates: CoordinateSystem,
        n_scan: int,
        n_modes: int = 1,
        mode_data_or_initializer: bool = None,
        weight_initializer: Callable[[tuple[int, ...]], np.ndarray] = default_weight_initializer,
        regularization_functions: tuple[Callable[[Complex[Array, "m n 1"]], float], ...] = (),
        *,
        key: jax.random.PRNGKey,
    ) -> None:
        if mode_data_or_initializer is None:  # TODO fix
            mode_keys = jax.random.split(key, n_modes)
            mode_data_or_initializer = functools.partial(
                default_mode_initializer,
                shape=probe_grid.shape,
                max_index=n_modes,
                wavelength=wavelength,
                probe_grid=probe_grid,
                coordinates=coordinates,
            )
        else:
            mode_keys = None
        self.n_modes = n_modes
        self.n_scan = n_scan
        if callable(mode_data_or_initializer):
            if mode_keys is not None:
                self.probe_modes = vmap(lambda i, k: mode_data_or_initializer(mode_index=i, key=k))(
                    jnp.arange(n_modes), mode_keys
                )
            else:
                self.probe_modes = vmap(mode_data_or_initializer)(jnp.arange(n_modes))
        else:
            self.probe_modes = vmap(lambda _: mode_data_or_initializer)(jnp.arange(n_modes))
        # assert self.probe_modes().shape[0] == self.N_modes
        self.singular_values = weight_initializer((self.n_scan, self.n_modes)) * jnp.sqrt(
            probe_grid.shape[0] * probe_grid.shape[1]
        )
        if self.singular_values.dtype.kind != "c":
            logging.warning(
                "Initializer for orthogonalized probe weights was not complex. Repeating the initializer for the "
                "imaginary part..."
            )
            self.singular_values = self.singular_values + 1j * self.singular_values
        self.singular_values = IndexSliceParameter(self.singular_values)
        self.regularization_functions = regularization_functions

    def __call__(self, *args, **kwargs) -> CoherentField:
        """Compute the effective probe by combining orthogonalized modes.

        Blends the raw *probe_modes* with their QR-orthogonalized versions
        according to *relaxed_orthogonalization*, then contracts with the
        per-scan *singular_values* to produce the output probe field.

        Returns:
            A :py:class:`~ptyrax.field.CoherentField` representing the
            combined probe for the current scan index.
        """
        probe_data = vmap(lambda probe: probe)(self.probe_modes)
        nearly_orthogonal_probes = probe_data * self.relaxed_orthogonalization + self.orthogonal_probes * (
            1 - self.relaxed_orthogonalization
        )

        current_weights = self.singular_values
        output_probe_data = jnp.einsum("d ..., d -> ...", nearly_orthogonal_probes, current_weights)
        first_probe = jax.tree.map(lambda leaf: leaf[0], self.probe_modes)
        output_probe = CoherentField(
            output_probe_data,
            first_probe.wavelength,
            first_probe.sampling,
            first_probe.coordinate_system,
            first_probe.propagation_direction,
            first_probe.spatial_dims,
            first_probe.vector_dim,
        )
        return output_probe

    @property
    def orthogonal_probes(self) -> Float[Array, "* m n d"]:
        """Orthogonalized probe modes obtained via QR decomposition.

        Flattens the spatial dimensions of the stored probe modes, computes a
        QR factorization, and reshapes the Q factor back to the original
        spatial dimensions to yield an orthonormal set of probe modes.

        Returns:
            Array of orthogonalized probe mode data with the same shape as
            the raw probe modes.
        """
        probe_data = vmap(lambda probe: probe)(self.probe_modes)
        original_shape = probe_data.shape
        flattened_data = probe_data.reshape((*probe_data.shape[:-3], -1))  # Flatten the spatial part
        Q, _ = jnp.linalg.qr(flattened_data.T)  # noqa: N806
        orthogonal_probes = Q.T.reshape(original_shape)
        return orthogonal_probes

    def __plot__(self, *args, **kwargs) -> tuple[Figure, SubplotSpec, Union[Array, list[Array]]]:
        """Plot the orthogonalized probe modes weighted by their singular
        values.

        Returns:
            Tuple of (figure, subplot_spec, plotted_data) as expected by the
            plotting framework.
        """
        return plot(
            self.orthogonal_probes[..., 0] * self.singular_values[0, :, np.newaxis, np.newaxis],
            *args,
            **kwargs,
        )

    def __regularize__(self) -> tuple[float, list[NamedLoss]]:
        """Compute regularization losses over the orthogonalized probe modes.

        Applies each function in *regularization_functions* to the
        orthogonalized probes and aggregates the results.

        Returns:
            A tuple of (total_loss, named_losses) with per-function, per-mode
            :py:class:`~ptyrax.models.illumination.NamedLoss` entries.
        """
        first_probe = jax.tree.map(lambda leaf: leaf[0], self.probe_modes)
        output_probe = CoherentField(
            self.orthogonal_probes,
            first_probe.wavelength,
            first_probe.sampling,
            first_probe.coordinate_system,
            first_probe.propagation_direction,
            first_probe.spatial_dims,
            first_probe.vector_dim,
        )
        regularizations = []
        total = 0.0
        for fn in self.regularization_functions:
            regularization_values = vmap(fn)(output_probe())
            total += jnp.sum(regularization_values)
            probe_indices = jnp.arange(len(regularization_values))
            regularizations.extend(
                NamedLoss(f"probe.{fn.__name__}.{mode_index}", reg_value)
                for mode_index, reg_value in zip(probe_indices, regularization_values)
            )
        return total, regularizations

    def __log_epoch__(self, writer: SummaryWriter, epoch: int, prefix: str = "", **kwargs) -> None:
        """Log orthogonalized probe mode images for the current epoch.

        Args:
            writer: TensorBoard SummaryWriter instance.
            epoch: Current training epoch number.
            prefix: Optional prefix for the log tag.
            **kwargs: Additional keyword arguments (unused).
        """
        logging.debug(f"orthogonal_probe_shapes: {self.orthogonal_probes.shape}")
        orthogonal_probes = self.orthogonal_probes
        spatial_shape = self.orthogonal_probes.shape[-3:-1]
        log_image(
            writer,
            "0/2:probe/image",
            orthogonal_probes.reshape((-1, *spatial_shape)),
            epoch,
        )


class PupilIllumination(DirectIllumination):
    """Illumination model exposing the pupil representation of a
    `DirectIllumination` probe.

    Computes the pupil via an inverse Fourier transform of the probe and
    provides methods to obtain propagated fields from the pupil
    representation.
    """

    pupil: CoherentField

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the pupil illumination model.

        Constructs the parent
        :py:class:`~ptyrax.models.illumination.DirectIllumination` and then
        computes the pupil field via an inverse Fourier transform of the
        stored probe.

        Args:
            *args: Positional arguments forwarded to
                :py:class:`~ptyrax.models.illumination.DirectIllumination`.
            **kwargs: Keyword arguments forwarded to
                :py:class:`~ptyrax.models.illumination.DirectIllumination`.
        """
        super().__init__(*args, **kwargs)
        pupil_data = jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.ifftshift(self.probe())))
        self.pupil = CoherentField(
            pupil_data,
            self._probe.wavelength,
            self._probe.sampling.to_far_field(self._probe.wavelength, 1),
            self._probe.coordinate_system,
            self._probe.propagation_direction,
            self._probe.spatial_dims,
            self._probe.vector_dim,
        )

    def __call__(self, *args, **kwargs) -> CoherentField:
        """Compute the probe field from the pupil via a forward Fourier
        transform.

        Returns:
            A :py:class:`~ptyrax.field.CoherentField` in the sample plane
            obtained by Fourier-transforming the stored pupil.
        """
        output_samples = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(self.pupil())))
        output_field = CoherentField(
            output_samples,
            self.pupil.wavelength,
            self.pupil.sampling.to_far_field(self.pupil.wavelength, 1),
            self.pupil.coordinate_system,
            self.pupil.propagation_direction,
            self.pupil.spatial_dims,
            self.pupil.vector_dim,
        )
        return output_field

    @property
    def probe(self, propagation_distance: Float[Array, ""] = None) -> CoherentField:
        r"""Compute the probe field from the pupil, optionally with defocus
        propagation.

        When *propagation_distance* is provided, a quadratic phase factor is
        applied to the pupil before Fourier-transforming, effectively
        propagating the probe by that distance.

        Args:
            propagation_distance: Optional scalar propagation distance.  If
                ``None``, no defocus is applied.

        Returns:
            A :py:class:`~ptyrax.field.CoherentField` representing the probe
            in the sample plane.
        """
        pupil_data = self.pupil.data
        if propagation_distance is not None:
            propagation_distance /= self.pupil.wavelength
            quadratic_phase = phase_only_exp(2 * np.pi * propagation_distance * jnp.sqrt(1 - self.pupil.sampling.rr))
            pupil_data = pupil_data * quadratic_phase[..., jnp.newaxis]

        output_samples = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(pupil_data)))

        output_field = CoherentField(
            output_samples,
            self.pupil.wavelength,
            self.pupil.sampling.to_far_field(self.pupil.wavelength, 1),
            self.pupil.coordinate_system,
            self.pupil.propagation_direction,
            self.pupil.spatial_dims,
            self.pupil.vector_dim,
        )
        return output_field
