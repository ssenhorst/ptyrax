from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import equinox as eqx
import gin
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.ndimage import map_coordinates
from jaxtyping import Array, Bool, Complex, Float

from ptyrax.field import CoherentField
from ptyrax.spatial import (
    CoordinateSystem,
    SamplingGrid,
    angle_safe,
    convert_coordinates_into_indices,
)
from ptyrax.utils import phase_only_exp


@gin.configurable()
class Propagator(eqx.Module):
    r"""Abstract propagator interface mapping input fields to a target geometry.

    A propagator models how the exit field from a sample propagates to the
    detector plane. Subclasses implement specific propagation physics (e.g.
    Fraunhofer far-field diffraction, Fresnel near-field propagation).

    Example:
        >>> propagator = FarfieldPropagator()
        >>> output_field, valid_mask = propagator(input_field, detector_coords, detector_grid)
    """

    @abstractmethod
    def __call__(
        self,
        input_field: CoherentField,
        output_coordinates: CoordinateSystem,
        output_grid: SamplingGrid,
    ) -> tuple[CoherentField, Bool[Array, " d"]]:
        """Propagate an input field to the output (detector) geometry.

        Args:
            input_field: The coherent field at the sample exit plane.
            output_coordinates: The coordinate system of the detector,
                including its position and orientation relative to the
                global frame.
            output_grid: The sampling grid defining detector pixel
                positions.

        Returns:
            A tuple of:
                - The propagated coherent field at the detector plane.
                - A boolean mask indicating which detector pixels receive
                  valid signal (pixels outside the Ewald sphere coverage
                  are marked False).
        """
        pass

    @staticmethod
    @abstractmethod
    def input_coordinates(
        output_coordinates: CoordinateSystem,
        output_grid: SamplingGrid,
        wavelength: float,
    ) -> tuple[CoordinateSystem, SamplingGrid]:
        """Compute the input (sample-side) coordinates from detector geometry.

        Given the detector coordinate system and sampling grid, determines the
        corresponding coordinate system and sampling grid at the sample exit
        plane. This is used to initialize the probe and sample grids
        consistently with the detector geometry.

        Args:
            output_coordinates: The coordinate system of the detector.
            output_grid: The sampling grid of the detector.
            wavelength: The illumination wavelength in meters.

        Returns:
            A tuple of:
                - The coordinate system at the sample exit plane.
                - The sampling grid at the sample exit plane, derived from
                  the far-field reciprocal relationship.
        """
        pass


def _propagate_fresnel(field: CoherentField, target_coordinate_system: CoordinateSystem, *args) -> Array:
    xx, yy = field.sampling.meshgrid
    propagation_distance = jnp.linalg.norm(target_coordinate_system.translation)
    quadratic_phase = phase_only_exp(2 * np.pi * 1 / (2 * field.wavelength * propagation_distance) * (xx**2 + yy**2))
    return jnp.fft.fftshift(
        jnp.fft.fft2(
            jnp.fft.fftshift((field() * quadratic_phase[..., jnp.newaxis])),
            axes=field.spatial_dims,
            norm="ortho",
        )
    )


def _propagate_fresnel_nograd(field: CoherentField, target_coordinate_system: CoordinateSystem, *args) -> Array:
    xx, yy = field.sampling.meshgrid
    propagation_distance = jnp.linalg.norm(target_coordinate_system.translation)
    quadratic_phase = phase_only_exp(2 * np.pi * 1 / (2 * field.wavelength * propagation_distance) * (xx**2 + yy**2))
    quadratic_phase = lax.stop_gradient(quadratic_phase)
    return jnp.fft.fftshift(
        jnp.fft.fft2(
            jnp.fft.fftshift((field() * quadratic_phase[..., jnp.newaxis])),
            axes=field.spatial_dims,
            norm="ortho",
        )
    )


def _propagate_fraunhofer(field: CoherentField, target_coordinate_system: CoordinateSystem, *args) -> Array:
    return jnp.fft.fftshift(
        jnp.fft.fft2(
            jnp.fft.fftshift(field()),
            axes=field.spatial_dims,
            norm="ortho",
        )
    )


def nearfield_propagation_coefficient_fourier(
    k_z: Float[Array, "m n"],
    z_distance: float,
    valid: Bool[Array, "m n"] = None,
) -> Complex[Array, "m n"]:
    phase = 2 * jnp.pi * k_z * z_distance
    coefficient = jnp.exp(-1j * phase)
    if valid is not None:
        coefficient = jnp.where(valid, coefficient, 0.0)
    return coefficient


def _interpolate(
    samples: Float[Array, "* m n"],
    indices: Float[Array, "d 2"],
    interpolation_mode: Literal["real_imaginary", "amplitude_phase"] = "real_imaginary",
) -> Float[Array, " d"]:
    if interpolation_mode == "real_imaginary":
        interpolated_output_samples = map_coordinates(
            samples,
            indices,
            order=1,
            mode="constant",
        )
    elif interpolation_mode == "amplitude_phase":
        interpolated_output_samples_amp = map_coordinates(
            jnp.abs(samples),
            indices,
            order=1,
            mode="constant",
        )
        interpolated_output_samples_all = map_coordinates(
            samples,
            indices,
            order=1,
            mode="constant",
        )
        interpolated_output_samples = interpolated_output_samples_amp * phase_only_exp(
            angle_safe(interpolated_output_samples_all)
        )
    else:
        raise ValueError('interpolation mode must be one of ["real_imaginary", "amplitude_phase"]')

    return interpolated_output_samples


@gin.configurable()
def propagate_tilted(
    field: CoherentField,
    target_coordinate_system: CoordinateSystem,
    target_sampling: SamplingGrid,
    interpolation_mode: Literal["real_imaginary", "amplitude_phase"] = "real_imaginary",
    propagator_type: Literal["fresnel", "fraunhofer", "fresnel_nograd"] = "fraunhofer",
    apply_jacobian: bool = False,
    skip_interpolate: bool = False,
) -> tuple[CoherentField, Bool[Array, " d"]]:
    r"""Propagate a coherent field to a tilted detector plane.

    Performs far-field or near-field propagation followed by interpolation
    onto the tilted detector grid. The detector may be arbitrarily oriented
    relative to the sample frame, as specified by ``target_coordinate_system``.

    The propagation involves:
        1. Computing the Fourier transform of the input field (Fraunhofer or
           Fresnel).
        2. Mapping the tilted detector pixels onto the Ewald sphere in the
           source Fourier frame.
        3. Interpolating the propagated field at the mapped positions.

    Args:
        field: The coherent exit field at the sample plane.
        target_coordinate_system: Coordinate system of the detector,
            defining its position and orientation.
        target_sampling: Sampling grid of the detector pixels.
        interpolation_mode: How to interpolate complex values.
            ``"real_imaginary"`` interpolates real and imaginary parts
            independently. ``"amplitude_phase"`` interpolates amplitude
            directly and uses the phase of the real/imaginary interpolation.
        propagator_type: The propagation model to use.
            ``"fraunhofer"`` for far-field, ``"fresnel"`` for near-field
            with quadratic phase, ``"fresnel_nograd"`` for Fresnel with
            the quadratic phase factor excluded from gradient computation.
        apply_jacobian: If True, applies the Jacobian correction factor
            from the Ewald sphere projection to preserve energy density.

    Returns:
        A tuple of:
            - The propagated :py:class:`~ptyrax.field.CoherentField` at the
              detector plane.
            - A boolean mask of shape ``target_sampling.shape`` where True
              indicates valid detector pixels covered by the source Fourier
              space.
    """

    propagator = (
        _propagate_fresnel
        if propagator_type == "fresnel"
        else _propagate_fraunhofer
        if propagator_type == "fraunhofer"
        else _propagate_fresnel_nograd
        if propagator_type == "fresnel_nograd"
        else None
    )

    propagated_field_samples = propagator(field, target_coordinate_system)

    if skip_interpolate:
        interpolated_output_samples = propagated_field_samples
        mask = jnp.ones(target_sampling.shape, dtype=bool)
    else:
        field_frame_target_indices, mask, transform_jacobian_factor = tilt_interpolation_indices(
            field, target_coordinate_system, target_sampling
        )
        if apply_jacobian:
            propagated_field_samples = propagated_field_samples * transform_jacobian_factor
        interpolated_output_samples = _interpolate(
            propagated_field_samples[..., 0],
            field_frame_target_indices,
            interpolation_mode,
        )
        interpolated_output_samples = interpolated_output_samples.reshape((*target_sampling.shape, 1))
    mask = mask.reshape(target_sampling.shape)

    output_field = CoherentField(
        interpolated_output_samples,
        field.wavelength,
        target_sampling,
        target_coordinate_system,
        field.propagation_direction,
        field.spatial_dims,
        field.vector_dim,
    )
    return output_field, mask


def tilt_interpolation_indices(
    field: CoherentField,
    target_coordinate_system: CoordinateSystem,
    target_sampling: SamplingGrid,
) -> tuple[Float[Array, "d 2"], Bool[Array, " d"], Float[Array, " d"]]:
    r"""Compute interpolation indices for mapping a tilted detector onto the
    source Fourier grid.

    Maps detector pixel positions through the Ewald sphere geometry to
    determine which source Fourier-space pixels they correspond to. This
    enables interpolation of the propagated field onto an arbitrarily
    oriented detector in the far-field.

    Args:
        field: The coherent field whose Fourier grid defines the source
            frame.
        target_coordinate_system: Coordinate system of the tilted
            detector.
        target_sampling: Sampling grid of the detector pixels.

    Returns:
        A tuple of:
            - Interpolation indices of shape ``(2, d)`` where ``d`` is the
              total number of detector pixels. Each column gives the
              fractional (x, y) index into the source Fourier grid.
            - A boolean mask of shape ``(d,)`` indicating which detector
              pixels fall within the source Fourier grid bounds.
            - The Jacobian correction factor of shape ``(d,)`` for energy
              density preservation under the coordinate transformation.
    """
    farfield_sampling, field_frame_target_xy, transform_jacobian_factor = tilt_target_to_incident(
        field.sampling,
        field.coordinate_system,
        target_sampling,
        target_coordinate_system,
        field.propagation_direction,
        field.wavelength,
    )
    field_frame_target_indices, mask = convert_coordinates_into_indices(field_frame_target_xy, farfield_sampling)
    return field_frame_target_indices, mask, transform_jacobian_factor


def tilt_target_to_incident(
    sampling: SamplingGrid,
    coordinate_system: CoordinateSystem,
    target_sampling: SamplingGrid,
    target_coordinate_system: CoordinateSystem,
    illumination_direction: Float[Array, "3"],
    wavlen: float,
) -> tuple[SamplingGrid, Float[Array, "d 2"]]:
    r"""Transform detector coordinates into the incident field's Fourier frame.

    Computes the scattering vector for each detector pixel by projecting
    detector positions onto the Ewald sphere and subtracting the illumination
    direction. The resulting scattering vectors are then rotated into the
    source field's local coordinate frame.

    The scattering vector is defined as:

    .. math::

        \mathbf{q} = \hat{\mathbf{k}}_\text{out} - \hat{\mathbf{k}}_\text{in}

    where :math:`\hat{\mathbf{k}}_\text{out}` is the unit vector from sample
    to detector pixel (Ewald sphere projection) and
    :math:`\hat{\mathbf{k}}_\text{in}` is the unit illumination direction.

    Args:
        sampling: The sampling grid of the source field.
        coordinate_system: The coordinate system of the source field.
        target_sampling: The sampling grid of the detector.
        target_coordinate_system: The coordinate system of the detector.
        illumination_direction: The illumination wave vector direction
            as a 3D vector (not necessarily unit length).
        wavlen: The illumination wavelength in meters.

    Returns:
        A tuple of:
            - The far-field sampling grid corresponding to the source field.
            - The (x, y) coordinates of each detector pixel in the source
              field's Fourier frame, shape ``(2, d)``.
            - The Jacobian correction factor (z-component of Ewald sphere
              unit vectors) of shape ``(d,)``.
    """
    detector_coordinates_sphere = detector_sphere_coordinates(target_coordinate_system, target_sampling)
    transform_jacobian_factor = detector_coordinates_sphere[-1]

    illumination_coordinate_sphere = illumination_direction / jnp.linalg.norm(illumination_direction)

    global_frame_scattering_vector = detector_coordinates_sphere - illumination_coordinate_sphere

    farfield_sampling = sampling.to_far_field(wavlen, 1)
    field_frame_scattering_vector = jnp.einsum(
        "id, nd -> ni",
        coordinate_system.rotation.as_matrix(),
        global_frame_scattering_vector,
    )
    field_frame_target_xy = field_frame_scattering_vector[..., 0:2]
    field_frame_target_xy = jnp.moveaxis(field_frame_target_xy, -1, 0)
    return farfield_sampling, field_frame_target_xy, transform_jacobian_factor


def source_fourier_occupancy_from_tilt_interpolation(
    field: CoherentField,
    target_coordinate_system: CoordinateSystem,
    target_sampling: SamplingGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute which source Fourier pixels are actually observed by the tilted
    detector.

    Determines the set of source Fourier-space pixels that are "hit" by at
    least one detector pixel when mapped through the Ewald sphere geometry.
    This is useful for understanding which spatial frequencies of the sample
    are accessible given the detector placement.

    Args:
        field: The coherent field defining the source Fourier grid.
        target_coordinate_system: Coordinate system of the tilted
            detector.
        target_sampling: Sampling grid of the detector.

    Returns:
        A tuple of:
            - ``source_occupancy``: A boolean array of shape
              ``field.sampling.shape`` where True indicates that the
              corresponding Fourier pixel is observed by at least one
              detector pixel.
            - ``detector_mask``: A boolean array of shape
              ``target_sampling.shape`` indicating valid detector pixels.
            - ``field_frame_target_indices``: The interpolation indices
              array of shape ``(2, d)`` mapping detector pixels to source
              Fourier indices.
    """
    field_frame_target_indices, detector_mask_flat, _ = tilt_interpolation_indices(
        field,
        target_coordinate_system,
        target_sampling,
    )
    detector_mask = np.asarray(detector_mask_flat).reshape(target_sampling.shape)

    n_x, n_y = field.sampling.shape
    i_x = np.asarray(field_frame_target_indices[0])
    i_y = np.asarray(field_frame_target_indices[1])
    valid = np.asarray(detector_mask_flat)

    i_x = i_x[valid]
    i_y = i_y[valid]
    i_x_nn = np.rint(i_x).astype(int)
    i_y_nn = np.rint(i_y).astype(int)
    in_bounds = (i_x_nn >= 0) & (i_x_nn < n_x) & (i_y_nn >= 0) & (i_y_nn < n_y)

    source_occupancy = np.zeros((n_x, n_y), dtype=bool)
    source_occupancy[i_x_nn[in_bounds], i_y_nn[in_bounds]] = True
    return source_occupancy, detector_mask, np.asarray(field_frame_target_indices)


def _points_in_polygon(points: np.ndarray, vertices: np.ndarray, radius: float = 0.0) -> np.ndarray:
    """Test whether points lie inside a polygon using the winding number
    algorithm.

    Args:
        points: Array of shape ``(n, 2)`` with query points.
        vertices: Array of shape ``(m, 2)`` with polygon vertices (ordered).
        radius: Expand the polygon boundary by this amount (approximate,
            achieved by offsetting edges outward).

    Returns:
        Boolean array of shape ``(n,)`` indicating containment.
    """
    n_verts = len(vertices)
    winding = np.zeros(len(points), dtype=int)

    for i in range(n_verts):
        v0 = vertices[i]
        v1 = vertices[(i + 1) % n_verts]

        # Optionally offset edge outward by radius
        if radius != 0.0:
            edge = v1 - v0
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm * radius
                v0 = v0 - normal
                v1 = v1 - normal

        y0 = v0[1] - points[:, 1]
        y1 = v1[1] - points[:, 1]

        # Upward crossing
        up = (y0 <= 0) & (y1 > 0)
        # Downward crossing
        down = (y0 > 0) & (y1 <= 0)

        # Compute x-position of crossing
        cross = (v1[0] - v0[0]) * (-y0) / (y1 - y0 + 1e-30) + v0[0] - points[:, 0]

        winding += up & (cross > 0)
        winding -= down & (cross > 0)

    return winding != 0


def source_fourier_support_from_tilt_interpolation(
    field: CoherentField,
    target_coordinate_system: CoordinateSystem,
    target_sampling: SamplingGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the filled Fourier support mask from tilted detector coverage.

    Extends :py:func:`~ptyrax.models.propagation.source_fourier_occupancy_from_tilt_interpolation`
    by constructing a convex polygon from the detector boundary pixels
    and filling the interior to produce a continuous support mask. This
    accounts for the fact that not every interior Fourier pixel may be
    directly hit by a detector pixel due to discrete sampling, but is
    still within the accessible frequency range.

    Args:
        field: The coherent field defining the source Fourier grid.
        target_coordinate_system: Coordinate system of the tilted
            detector.
        target_sampling: Sampling grid of the detector.

    Returns:
        A tuple of:
            - ``source_support``: A boolean array of shape
              ``field.sampling.shape`` representing the filled Fourier
              support region (union of polygon interior and direct
              occupancy).
            - ``source_occupancy``: A boolean array of directly occupied
              Fourier pixels (same as from
              :py:func:`~ptyrax.models.propagation.source_fourier_occupancy_from_tilt_interpolation`).
            - ``detector_mask``: A boolean array of valid detector pixels.
            - ``field_frame_target_indices``: The interpolation indices
              array of shape ``(2, d)``.
    """
    source_occupancy, detector_mask, field_frame_target_indices = source_fourier_occupancy_from_tilt_interpolation(
        field,
        target_coordinate_system,
        target_sampling,
    )

    n_x, n_y = field.sampling.shape
    i_x = np.asarray(field_frame_target_indices[0]).reshape(target_sampling.shape)
    i_y = np.asarray(field_frame_target_indices[1]).reshape(target_sampling.shape)

    m, n = target_sampling.shape
    if m < 2 or n < 2:
        return source_occupancy.copy(), source_occupancy, detector_mask, np.asarray(field_frame_target_indices)

    boundary_row = np.concatenate(
        [
            np.zeros(n, dtype=int),
            np.arange(1, m, dtype=int),
            np.full(n - 1, m - 1, dtype=int),
            np.arange(m - 2, 0, -1, dtype=int),
        ]
    )
    boundary_col = np.concatenate(
        [
            np.arange(n, dtype=int),
            np.full(m - 1, n - 1, dtype=int),
            np.arange(n - 2, -1, -1, dtype=int),
            np.zeros(m - 2, dtype=int),
        ]
    )

    boundary_valid = detector_mask[boundary_row, boundary_col]
    boundary_x = i_x[boundary_row, boundary_col][boundary_valid]
    boundary_y = i_y[boundary_row, boundary_col][boundary_valid]

    finite = np.isfinite(boundary_x) & np.isfinite(boundary_y)
    boundary_x = boundary_x[finite]
    boundary_y = boundary_y[finite]

    if boundary_x.size < 3:
        return source_occupancy.copy(), source_occupancy, detector_mask, np.asarray(field_frame_target_indices)

    vertices = np.stack([boundary_x, boundary_y], axis=-1)

    grid_x, grid_y = np.meshgrid(np.arange(n_x, dtype=float), np.arange(n_y, dtype=float), indexing="ij")
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    source_support = _points_in_polygon(grid_points, vertices, radius=0.5).reshape((n_x, n_y))
    source_support = source_support | source_occupancy

    return source_support, source_occupancy, detector_mask, np.asarray(field_frame_target_indices)


def detector_sphere_coordinates(
    target_coordinate_system: CoordinateSystem,
    target_sampling: SamplingGrid,
) -> Float[Array, "n 3"]:
    r"""Compute unit vectors on the Ewald sphere for each detector pixel.

    Projects each detector pixel position into 3D global coordinates using
    the detector's coordinate system (position and orientation), then
    normalizes to obtain unit vectors pointing from the sample to each
    pixel. These unit vectors lie on the Ewald sphere (when multiplied by the wavenumber) and represent the
    outgoing wavevector directions :math:`\hat{\mathbf{k}}_\text{out}`.

    Args:
        target_coordinate_system: Coordinate system of the detector,
            defining its translation (distance from sample) and rotation
            (orientation).
        target_sampling: Sampling grid defining the physical positions
            of detector pixels.

    Returns:
        An array of shape ``(n, 3)`` containing the unit vectors for each
        of the ``n`` detector pixels in global coordinates.
    """
    detector_meshgrid = target_sampling.meshgrid.reshape((2, -1))
    detector_meshgrid = jnp.moveaxis(detector_meshgrid, 0, -1)
    detector_meshgrid = jnp.append(detector_meshgrid, jnp.zeros_like(detector_meshgrid[:, 0:1]), axis=-1)

    target_rotation_matrix = target_coordinate_system.rotation.as_matrix()
    detector_coordinates = target_coordinate_system.translation[jnp.newaxis, :] + jnp.einsum(
        "dj, nj -> nd",
        target_rotation_matrix.T,
        detector_meshgrid,
    )
    return detector_coordinates / jnp.linalg.norm(detector_coordinates, axis=-1, keepdims=True)


@gin.configurable()
class FarfieldPropagator(Propagator):
    r"""Far-field (Fraunhofer) propagator for ptychographic reconstruction.

    Implements propagation from the sample exit plane to a detector in the
    far-field regime using Fraunhofer diffraction. The detected field is
    the Fourier transform of the exit field, mapped onto the detector
    through the Ewald sphere geometry.

    This propagator supports arbitrarily tilted detectors (e.g. for
    reflection-geometry ptychography) by delegating to
    :py:func:`~ptyrax.models.propagation.propagate_tilted`.

    Example:
        >>> propagator = FarfieldPropagator()
        >>> detected_field, mask = propagator(exit_field, detector_coords, detector_grid)
    """

    def __call__(
        self,
        input_field: CoherentField,
        output_coordinates: CoordinateSystem,
        output_grid: SamplingGrid,
    ) -> tuple[CoherentField, Bool[Array, " d"]]:
        """Propagate an input field to the far-field detector plane.

        Applies Fraunhofer propagation with tilt correction to map the
        exit field onto the detector grid.

        Args:
            input_field: The coherent exit field at the sample plane.
            output_coordinates: Coordinate system of the detector.
            output_grid: Sampling grid of the detector pixels.

        Returns:
            A tuple of:
                - The propagated field at the detector plane.
                - A boolean mask indicating valid detector pixels.
        """
        return propagate_tilted(input_field, output_coordinates, output_grid)

    def input_coordinates(
        self,
        output_coordinates: CoordinateSystem,
        output_grid: SamplingGrid,
        wavelength: float,
    ) -> tuple[CoordinateSystem, SamplingGrid]:
        """Compute the sample-plane coordinates from detector geometry.

        Derives the input (sample-side) coordinate system and sampling
        grid from the detector configuration using the far-field reciprocal
        relationship. The input grid pixel size is determined by the
        wavelength and propagation distance.

        Args:
            output_coordinates: Coordinate system of the detector.
            output_grid: Sampling grid of the detector.
            wavelength: The illumination wavelength in meters.

        Returns:
            A tuple of:
                - The coordinate system at the sample plane (same
                  rotation as detector, zero translation).
                - The sampling grid at the sample plane, computed via
                  the far-field inverse relationship.
        """
        input_coordinates = CoordinateSystem(
            rotation=output_coordinates.rotation,
            translation=jnp.zeros_like(output_coordinates.translation),
        )
        propagation_distances = jnp.linalg.norm(output_coordinates.translation, axis=-1, keepdims=True)

        input_grids = eqx.filter_vmap(output_grid.to_far_field, in_axes=(None, 0))(wavelength, propagation_distances)

        return (input_coordinates, input_grids)
