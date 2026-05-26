from __future__ import annotations

import warnings
from typing import Callable, Self

import equinox as eqx
import gin
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Bool, Complex, Float, PRNGKeyArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.image import AxesImage
from tensorboardX import SummaryWriter

from ptyrax.initializers import gaussian
from ptyrax.logger import log_image
from ptyrax.parametrizations import ArrayParametrization
from ptyrax.spatial import CoordinateSystem, SamplingGrid, interpolate_grid_to_grid
from ptyrax.utils import fft, ifft, plot

X_AXIS = jnp.array([1.0, 0.0, 0.0])
Y_AXIS = jnp.array([0.0, 1.0, 0.0])
Z_AXIS = jnp.array([0.0, 0.0, 1.0])


@gin.configurable
class CoherentField(eqx.Module):
    """Representation of a coherent field (probe or object) with sampling and
    wavelength metadata."""

    data: Float[Array, "* m n d"] | ArrayParametrization

    wavelength: Float[Array, ""] = eqx.field(converter=lambda x: jnp.array(x))

    sampling: SamplingGrid

    coordinate_system: CoordinateSystem = eqx.field(default_factory=lambda: CoordinateSystem())

    propagation_direction: Array = eqx.field(default_factory=lambda: Z_AXIS.copy())

    spatial_dims: tuple = eqx.field(static=True, default=(-2, -3))

    vector_dim: int = eqx.field(static=True, default=-1)

    def __call__(self) -> Float[Array, "* m n d"]:
        """Return the underlying field data, resolving any parametrization.

        If the data is wrapped in an :py:class:`~ptyrax.parametrizations.ArrayParametrization`,
        this evaluates it to produce the raw array. Otherwise returns the data directly.

        Returns:
            The complex-valued field array.
        """
        if isinstance(self.data, ArrayParametrization):
            return self.data()
        return self.data

    @property
    def _data(self) -> Float[Array, "* m n d"]:
        warnings.warn("Using _data is deprecated. Use the .data property instead.", DeprecationWarning)
        return self.data

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        return self.data.shape

    def _apply_fftshift(self, target_shifted: bool) -> CoherentField:
        """Apply or undo FFT-shift along spatial dimensions.

        Args:
            target_shifted: Whether the result should be FFT-shifted.

        Returns:
            A new :py:class:`~ptyrax.field.CoherentField` with (un)shifted data
            and updated sampling metadata.
        """
        if self.sampling.fftshifted == target_shifted:
            return self
        shift_fn = jnp.fft.fftshift if target_shifted else jnp.fft.ifftshift
        new_data = shift_fn(self.data, axes=self.spatial_dims)
        new_data = jnp.asarray(new_data) if isinstance(self.data, jnp.ndarray) else type(self.data)(new_data)
        new_sampling = SamplingGrid(
            self.sampling.shape,
            self.sampling.pixel_size,
            self.sampling.origin_shift,
            fftshifted=target_shifted,
        )
        return CoherentField(
            new_data,
            self.wavelength,
            new_sampling,
            self.coordinate_system,
            self.propagation_direction,
            self.spatial_dims,
            self.vector_dim,
        )

    @property
    def to_fft_shifted(self) -> CoherentField:
        """Return an FFT-shifted version of this field.

        Applies ``jnp.fft.fftshift`` along the spatial dimensions so that the
        zero-frequency component is centered.  If the field is already shifted,
        returns ``self`` unchanged.

        Returns:
            A new :py:class:`~ptyrax.field.CoherentField` with shifted data and
            updated sampling metadata.
        """
        return self._apply_fftshift(target_shifted=True)

    @property
    def to_fft_unshifted(self) -> CoherentField:
        """Return a non-FFT-shifted version of this field.

        Applies ``jnp.fft.ifftshift`` along the spatial dimensions to restore
        the standard array ordering.  If the field is already unshifted, returns
        ``self`` unchanged.

        Returns:
            A new :py:class:`~ptyrax.field.CoherentField` with unshifted data and
            updated sampling metadata.
        """
        return self._apply_fftshift(target_shifted=False)

    @property
    def amplitude(self) -> Float[Array, "* m n d"]:
        """Absolute value (amplitude) of the field, FFT-shifted if
        applicable."""
        amplitude = jnp.abs(self.data)
        if self.sampling.fftshifted:
            amplitude = jnp.fft.fftshift(amplitude, axes=self.spatial_dims)
        return amplitude

    @property
    def intensity(self) -> Float[Array, "* m n d"]:
        r"""Squared magnitude (intensity) of the field: :math:`|E|^2`.

        The result is real-valued.
        """
        intensity = self.data * self.data.conj()
        if self.sampling.fftshifted:
            intensity = jnp.fft.fftshift(intensity, axes=self.spatial_dims)
        return intensity.real

    def __getitem__(self, item: slice) -> Self:
        """Slice the field along its leading (mode/batch) dimensions.

        All sub-components (data, wavelength, sampling, coordinate_system,
        propagation_direction) are sliced consistently.

        Args:
            item: Index or slice to apply along the leading dimensions.

        Returns:
            A new :py:class:`~ptyrax.field.CoherentField` containing the
            selected subset.
        """
        return CoherentField(
            self.data[item],
            self.wavelength[item],
            self.sampling[item],
            self.coordinate_system[item],
            self.propagation_direction[item],
            self.spatial_dims,
            self.vector_dim,
        )

    def __plot__(
        self,
        show: bool = True,
        gs: SubplotSpec = None,
        fig: Figure = None,
        **kwargs,
    ) -> tuple[Figure, Axes, SubplotSpec]:
        """Visualize the field as amplitude and phase images.

        Uses the first vector component (index 0 along the last axis) and
        shows amplitude/phase in the same plot using a bivariate colormap via :py:func:`~ptyrax.utils.plot`.

        Args:
            show: Whether to call ``plt.show()`` after plotting.
            gs: Optional matplotlib ``SubplotSpec`` to draw into.
            fig: Optional matplotlib ``Figure`` to use.
            **kwargs: Additional keyword arguments forwarded to
                :py:func:`~ptyrax.utils.plot`.

        Returns:
            A tuple of (Figure, Axes, SubplotSpec).
        """
        if gs is None:
            fig = plt.figure()
            gs = fig.add_gridspec(1, 1)[0]
        data = self.data
        if self.sampling.fftshifted:
            data = np.fft.fftshift(data, axes=(0, 1))
        extent = self.sampling.extent
        return plot(data[..., 0], show=show, extent=extent, gs=gs, fig=fig, **kwargs)

    def __log_epoch__(self, writer: SummaryWriter, epoch: int, prefix: str = "", **kwargs) -> None:
        """Log field state to TensorBoard at a given epoch.

        Logs per-mode wavelength scalars, a probe image, and the field norm.

        Args:
            writer: TensorBoardX summary writer instance.
            epoch: Current training epoch number.
            prefix: Optional string prefix for logged tags.
            **kwargs: Additional keyword arguments (unused).
        """
        for i, wavelength in enumerate(self.wavelength):
            writer.add_scalar(f"2_probe/{prefix}wavelength/{i}", wavelength, epoch)
        log_image(writer, f"0/2_probe/{prefix}image", self, epoch, title="probe")
        writer.add_scalar(f"2_probe/{prefix}norm", jnp.linalg.norm(self()), epoch)

    def propagate_fraunhofer(self, distance: float, inverse: bool = False, fftshift: bool = True) -> CoherentField:
        r"""Propagate the field to the far field using the Fraunhofer
        approximation.

        Computes the Fourier transform (or inverse) of the first vector
        component and updates the sampling grid to far-field coordinates.

        Args:
            distance: Propagation distance used to compute the far-field
                pixel size.
            inverse: If ``True``, use the inverse FFT (propagate back to
                near field).
            fftshift: Whether to apply ``fftshift`` to the result.

        Returns:
            A new :py:class:`~ptyrax.field.CoherentField` with propagated data
            and far-field sampling.
        """
        propagated_probe_data = (
            ifft(self()[..., 0], fftshift=fftshift) if inverse else fft(self()[..., 0], fftshift=fftshift)
        )
        propagated_probe_data = propagated_probe_data[..., None]

        propagated_probe_sampling = self.sampling.to_far_field(self.wavelength, distance, fftshifted=not fftshift)

        propagated_probe = eqx.tree_at(lambda p: p.data, self, propagated_probe_data)
        propagated_probe = eqx.tree_at(lambda p: p.sampling, propagated_probe, propagated_probe_sampling)
        return propagated_probe

    def propagate_tilted_nearfield(
        self, displacement: Float[Array, "3"], n_medium: complex = 1.0 + 0j
    ) -> CoherentField:
        r"""Propagate the field in the near-field regime along a tilted
        direction.

        Performs angular-spectrum propagation by applying a phase ramp in
        Fourier space corresponding to a 3-D displacement vector.  Evanescent
        components (those outside the Ewald sphere) are set to zero.

        Args:
            displacement: 3-D displacement vector ``[dx, dy, dz]`` in real-space
                units.  Only the z-component contributes to the propagation
                phase; the transverse components are currently ignored.

        Returns:
            A new :py:class:`~ptyrax.field.CoherentField` with propagated data.
        """
        from ptyrax.models.propagation import nearfield_propagation_coefficient_fourier

        pupil_xi_z, valid = self.k_z(n_medium)
        propagation_factor = nearfield_propagation_coefficient_fourier(
            k_z=pupil_xi_z,
            z_distance=displacement[-1],
            valid=valid,
        )
        propagated_probe = self.multiply_fourier(propagation_factor)

        return propagated_probe

    def k_z(self, n_medium: complex = 1.0 + 0j) -> tuple[Float[Array, "m n"], Bool[Array, "m n"]]:
        """Compute the z-component of the wave vector in the field's internal
        coordinate system."""
        pupil_sampling = self.sampling.to_far_field(
            wavelength=self.wavelength,
            propagation_distance=1.0,
            fftshifted=False,
        )
        pupil_meshgrid = pupil_sampling.meshgrid
        pupil_rotation_matrix = self.coordinate_system.rotation.as_matrix()
        internal_propagation_direction = pupil_rotation_matrix @ self.propagation_direction
        pupil_meshgrid = pupil_meshgrid + internal_propagation_direction[:2, None, None] / self.wavelength

        pupil_xi_z = jnp.sqrt(n_medium**2 / (self.wavelength) ** 2 - jnp.sum(pupil_meshgrid**2, axis=0))
        valid = ~jnp.isnan(pupil_xi_z) & (pupil_xi_z.imag == 0)
        pupil_xi_z = jnp.nan_to_num(pupil_xi_z, nan=0.0)
        return pupil_xi_z, valid

    def multiply_fourier(self, factor: Complex[Array, "m n"]) -> CoherentField:
        """Multiply the field by a Fourier-space factor.

        The factor is applied to the first vector component in Fourier space,
        and the result is transformed back to real space.

        Args:
            factor: Complex array of shape ``(m, n)`` representing the Fourier
                space factor to apply.
        Returns:
            A new :py:class:`~ptyrax.field.CoherentField` with the modified data.
        """
        pupil = self.propagate_fraunhofer(1.0, fftshift=True)
        modified_pupil_data = pupil()[..., 0] * factor
        modified_probe_data = ifft(modified_pupil_data, fftshift=True)
        modified_probe_data = modified_probe_data[..., None]
        modified_probe = eqx.tree_at(lambda p: p.data, self, modified_probe_data)
        return modified_probe

    def multiply_real(self, factor: Float[Array, "m n"]) -> CoherentField:
        """Multiply the field by a real-valued factor in real space."""
        modified_probe_data = self() * factor[..., None]
        modified_probe = eqx.tree_at(lambda p: p.data, self, modified_probe_data)
        return modified_probe

    def __add__(self, other: CoherentField) -> CoherentField:
        return eqx.tree_at(lambda p: p.data, self, self() + other())


@gin.configurable
def plot_field(field: CoherentField, show: bool = True, **kwargs) -> tuple[Figure, SubplotSpec, list[AxesImage]]:
    """Plot a coherent field showing amplitude and phase.

    Extracts the first mode and first vector component, applies FFT-shift if
    needed, and delegates to :py:func:`~ptyrax.utils.plot`.

    Args:
        field: The coherent field to visualize.
        show: Whether to call ``plt.show()`` after plotting.
        **kwargs: Additional keyword arguments forwarded to
            :py:func:`~ptyrax.utils.plot`.

    Returns:
        A tuple of (Figure, SubplotSpec, list of AxesImage).
    """
    data = field.flattened.data[0, :, :, 0]
    if field.sampling.fftshifted:
        data = np.fft.fftshift(data)
    fov = field.flattened.field_of_view
    extent = [jnp.squeeze(fov[0, 0]), jnp.squeeze(fov[0, 1]), jnp.squeeze(fov[1, 0]), jnp.squeeze(fov[1, 1])]
    return plot(data, show=show, extent=extent, **kwargs)


@gin.configurable
def initialize_new_probe(
    old_probe: CoherentField,
    new_initializer_functions: tuple[Callable[[tuple[int, int]], CoherentField]],
    *,
    key: PRNGKeyArray = jax.random.PRNGKey(0),
) -> CoherentField:
    """Create a new probe by reinitializing each mode with a given function.

    Each mode of ``old_probe`` is reinitialized using the corresponding
    initializer function.  If a single initializer is provided it is broadcast
    to all modes.

    Args:
        old_probe: The existing probe whose shape and metadata are preserved.
        new_initializer_functions: A tuple of callables, each accepting a shape
            tuple and a ``key`` keyword argument and returning an array for one
            mode.  A single-element tuple is broadcast to all modes.
        key: JAX PRNG key used for random initialization.

    Returns:
        A new :py:class:`~ptyrax.field.CoherentField` with reinitialized data.

    Raises:
        ValueError: If the number of initializer functions does not match the
            number of probe modes (and is not exactly one).
    """
    if len(new_initializer_functions) == 1:
        new_initializer_functions = new_initializer_functions * old_probe.shape[0]
    if len(new_initializer_functions) != old_probe.shape[0]:
        raise ValueError(
            f"Number of new initializers {len(new_initializer_functions)} does not match "
            f"number of probe modes {old_probe.shape[0]}"
        )

    def reinitialize_single_mode(
        old_mode: CoherentField,
        new_initializer_function_idx: int,
        *,
        key: PRNGKeyArray,
    ) -> CoherentField:
        initializer = new_initializer_functions[new_initializer_function_idx]
        new_mode_data = initializer(old_mode.shape[:-1], key=key)
        new_mode = eqx.tree_at(
            lambda p: p.data,
            old_mode,
            type(old_mode.data)(new_mode_data[None, ..., None]),
        )
        return new_mode

    reinitialize_modes = eqx.filter_vmap(reinitialize_single_mode, in_axes=(0, 0, 0))

    new_probe = reinitialize_modes(
        old_probe,
        jnp.arange(len(new_initializer_functions)),
        jax.random.split(key, old_probe.shape[0]),
    )
    return new_probe


@gin.configurable
def transpose_probe(old_probe: CoherentField) -> CoherentField:
    """Swap the spatial axes of a probe field.

    Transposes the two spatial dimensions and interpolates onto a new
    :py:class:`~ptyrax.spatial.SamplingGrid` with swapped shape and pixel size.

    Args:
        old_probe: The probe field to transpose.

    Returns:
        A new :py:class:`~ptyrax.field.CoherentField` with transposed spatial
        dimensions.
    """
    new_probe_data = np.swapaxes(old_probe(), -1, -2)
    sampled_grid = old_probe.sampling
    transpose_grid = SamplingGrid.from_tuples(sampled_grid.shape[::-1], sampled_grid.pixel_size[0, ::-1])
    new_probe_data = eqx.filter_vmap(
        lambda p: interpolate_grid_to_grid(
            p[..., 0], transpose_grid, CoordinateSystem(), sampled_grid, CoordinateSystem()
        )
    )(new_probe_data)
    return eqx.tree_at(lambda p: p.data, old_probe, type(old_probe.data)(new_probe_data[None, ..., None]))


@gin.configurable
def remove_field_phase(old_probe: CoherentField) -> CoherentField:
    """Remove the phase of a field, keeping only the amplitude.

    Args:
        old_probe: The coherent field whose phase is to be removed.

    Returns:
        A new :py:class:`~ptyrax.field.CoherentField` with real-valued
        (amplitude-only) data.
    """
    probe_new = np.abs(old_probe())
    return eqx.tree_at(lambda p: p.data, old_probe, type(old_probe.data)(probe_new[None, ..., None]))


@gin.configurable
def replace_defocus(
    old_field: CoherentField,
    defocus_amount: float | tuple[float, float],
    axis: str = "xy",
) -> CoherentField:
    """Replace the field phase with a defocus along the specified axis or axes.

    Args:
        old_field: The coherent field to modify.
        defocus_amount: Defocus strength in meters. A single float for single-axis
            defocus, or a tuple ``(defocus_x, defocus_y)`` for ``axis="xy"``.
        axis: Which axis to defocus: ``"x"``, ``"y"``, or ``"xy"``.

    Returns:
        A new :py:class:`~ptyrax.field.CoherentField` with the specified
        defocus phase.

    Raises:
        ValueError: If ``axis`` is not one of ``"x"``, ``"y"``, or ``"xy"``.
    """
    if axis == "x":
        defocus_tuple = (defocus_amount, 0.0)
    elif axis == "y":
        defocus_tuple = (0.0, defocus_amount)
    elif axis == "xy":
        defocus_tuple = defocus_amount if isinstance(defocus_amount, tuple) else (defocus_amount, defocus_amount)
    else:
        raise ValueError(f"Invalid axis: {axis!r}. Must be 'x', 'y', or 'xy'.")
    return replace_field_phase(
        old_field,
        defocus_phase_generator(old_field, defocus_amount=defocus_tuple),
    )


@gin.configurable
def replace_defocus_x(old_field: CoherentField, defocus_amount: float) -> CoherentField:
    """Replace the field phase with a defocus along the x-axis only."""
    return replace_defocus(old_field, defocus_amount, axis="x")


@gin.configurable
def replace_defocus_y(old_field: CoherentField, defocus_amount: float) -> CoherentField:
    """Replace the field phase with a defocus along the y-axis only."""
    return replace_defocus(old_field, defocus_amount, axis="y")


@gin.configurable
def replace_defocus_xy(old_field: CoherentField, defocus_amount: tuple[float, float]) -> CoherentField:
    """Replace the field phase with independent defocus along both axes."""
    return replace_defocus(old_field, defocus_amount, axis="xy")


def defocus_phase_generator(old_field: CoherentField, defocus_amount: tuple[float, float]) -> Float[Array, "n m 1"]:
    r"""Generate a quadratic defocus phase profile for a field.

    Uses :py:func:`~ptyrax.initializers.gaussian` with a negative radius to
    produce a pure defocus (quadratic phase) without amplitude modulation.

    Args:
        old_field: The field whose sampling grid defines the coordinate space.
        defocus_amount: Tuple of ``(defocus_x, defocus_y)`` in meters.

    Returns:
        A complex-valued array of shape ``(n, m, 1)`` representing the defocus
        phase factor.
    """
    return gaussian(
        old_field.sampling,
        radius=(-1, -1),
        defocus=defocus_amount,
    )


def replace_field_phase(
    old_probe: CoherentField,
    phase_or_generator: Complex[Array, "n m 1"] | Callable[[CoherentField], Complex[Array, "n m 1"]] | CoherentField,
) -> CoherentField:
    """Replace the phase of a field with a new phase profile.

    The amplitude of the original field is preserved while its phase is
    replaced by the phase of the provided array, callable, or field.

    Args:
        old_probe: The coherent field whose phase will be replaced.
        phase_or_generator: The new phase source as either a complex array,
            a callable from :py:class:`~ptyrax.field.CoherentField` to a
            complex array, or another
            :py:class:`~ptyrax.field.CoherentField`.

    Returns:
        A new :py:class:`~ptyrax.field.CoherentField` with the original
        amplitude and the new phase.
    """
    if isinstance(phase_or_generator, CoherentField):
        phase = phase_or_generator()
    elif callable(phase_or_generator):
        phase = phase_or_generator(old_probe)
    else:
        phase = phase_or_generator
    if len(phase.shape) == 2:
        phase = phase[..., None]
    old_probe_data = old_probe()
    if len(old_probe_data.shape) == 4:
        phase = phase[None, ...]

    probe_new = np.abs(old_probe()) * phase / jnp.abs(phase)
    new_data = type(old_probe.data) if isinstance(old_probe.data, ArrayParametrization) else probe_new
    return eqx.tree_at(lambda p: p.data, old_probe, new_data)


def multiply_field_phase(
    old_probe: CoherentField,
    phase_or_generator: Complex[Array, "n m"] | Callable[[CoherentField], Complex[Array, "n m"]],
) -> CoherentField:
    """Multiply the field by a phase-only factor.

    Unlike :py:func:`replace_field_phase`, this adds phase on top of the
    existing field phase (and amplitude) by multiplying the field with
    a unit-amplitude complex factor.

    Args:
        old_probe: The coherent field to modify.
        phase_or_generator: Phase factor to multiply, provided as a complex
            array or as a callable that accepts a
            :py:class:`~ptyrax.field.CoherentField` and returns a complex
            array.

    Returns:
        A new :py:class:`~ptyrax.field.CoherentField` with the additional
        phase applied.
    """
    if callable(phase_or_generator):
        phase = phase_or_generator(old_probe)
    else:
        phase = phase_or_generator

    probe_new = old_probe() * phase / jnp.abs(phase)
    return eqx.tree_at(lambda p: p.data, old_probe, type(old_probe)(probe_new))


@gin.configurable
def set_probe_wavelength(
    old_probe: CoherentField, wavelength: float | list | tuple | np.ndarray | jnp.ndarray
) -> CoherentField:
    """Set the wavelength(s) of a probe field.

    If a single float is given it is broadcast to all modes.  An array of
    wavelengths must match the number of probe modes.

    Args:
        old_probe: The probe field to update.
        wavelength: Wavelength value(s) in meters.  Either a single float
            (applied to all modes) or a sequence matching the mode count.

    Returns:
        A new :py:class:`~ptyrax.field.CoherentField` with updated wavelength.

    Raises:
        ValueError: If the number of provided wavelengths does not match the
            number of probe modes.
    """
    if isinstance(wavelength, float):
        wavelength = jnp.array([wavelength] * old_probe.shape[0])
    else:
        wavelength = jnp.array(wavelength)
        if len(wavelength) != old_probe.shape[0]:
            raise ValueError(
                f"Number of new wavelengths {len(wavelength)} does not match number of probe modes {old_probe.shape[0]}"
            )

    return eqx.tree_at(lambda p: p.wavelength, old_probe, wavelength)
