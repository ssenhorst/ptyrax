from typing import Callable, Literal, Self

import equinox as eqx
import gin
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from jax.scipy.ndimage import map_coordinates
from jax.scipy.spatial.transform import Rotation as JaxRotation
from jaxtyping import Array, ArrayLike, Bool, Complex, Float, Shaped
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.lines import Line2D

from ptyrax.parametrizations import (
    ArrayParametrization,
    resolve_array_parametrizations,
)
from ptyrax.utils import make_or_reuse_axes


def matrix_to_six_dimensional_representation(rotation_matrix: Float[Array, "... 3 3"]) -> Float[Array, "... 6"]:
    """Convert a 3x3 rotation matrix to its 6D continuous representation.

    Uses the first two rows of the rotation matrix as the 6D representation,
    following the approach from Zhou et al. (NeurIPS 2019) for continuous
    rotation representations suitable for gradient-based optimization.

    Args:
        rotation_matrix: A shape ``(..., 3, 3)`` rotation matrix.

    Returns:
        A shape ``(..., 6)`` array containing the first two rows of the
        rotation matrix concatenated.

    See Also:
        :py:func:`~ptyrax.spatial.six_dimensional_representation_to_matrix`:
            The inverse operation.
    """
    # TODO convert to extension of scipy.spatial.Rotation class
    rotation_matrix = jnp.atleast_1d(rotation_matrix)
    original_shape = rotation_matrix.shape

    rotation_matrix = jnp.reshape(rotation_matrix, (-1, 3, 3))
    x = rotation_matrix[..., 0, :]
    y = rotation_matrix[..., 1, :]

    representation = jnp.concatenate((x, y), axis=-1)
    representation = jnp.reshape(representation, original_shape[:-2] + (6,))

    return representation


def six_dimensional_representation_to_matrix(representation: Float[Array, "... 6"]) -> Float[Array, "... 3 3"]:
    """Converts a 6D rotation representation to a 3x3 rotation matrix.

    Based on:
    "On the continuity of rotation representations in neural networks"
    Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li.
    Conference on Neural Information Processing Systems (NeurIPS) 2019.
    Args:
        representation: A shape (..., 6) array representing the rotation as a 6D vector.
    Returns:
        A shape (..., 3, 3) array representing the rotation as a 3x3 matrix.
    """
    representation = jnp.atleast_1d(representation)
    # original_shape = representation.shape

    x = representation[..., 0:3]
    y = representation[..., 3:6]

    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    z = jnp.cross(x, y)
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    y = jnp.cross(z, x)

    rotation_matrix = jnp.stack((x, y, z), axis=-2)
    return rotation_matrix


class Rotation(eqx.Module):
    """An equinox module wrapping a rotation using the 6D continuous
    representation.

    Stores rotations internally as a 6D vector (the first two rows of the rotation
    matrix), which provides a continuous, singularity-free parameterization suitable
    for gradient-based optimization. Based on Zhou et al. (NeurIPS 2019).

    Supports batched rotations via leading batch dimensions, composition via ``*``,
    inversion via the :py:attr:`inv` property, and indexing via ``[]``.

    Example:
        >>> import jax.numpy as jnp
        >>> from ptyrax.spatial import Rotation
        >>> rot = Rotation.from_matrix(jnp.eye(3))
        >>> rot.as_matrix()  # Returns identity matrix
    """

    _representation_6d: Float[Array, "* 6"] = eqx.field(
        default_factory=lambda: matrix_to_six_dimensional_representation(jnp.eye(3))
    )

    @classmethod
    def from_matrix(cls, matrix: Float[Array, "* 3 3"]) -> Self:
        """Construct a rotation from a 3x3 rotation matrix.

        Args:
            matrix: A shape ``(*, 3, 3)`` rotation matrix.

        Returns:
            A new :py:class:`~ptyrax.spatial.Rotation` instance.
        """
        return cls(matrix_to_six_dimensional_representation(matrix))

    def as_matrix(self) -> Float[Array, "* 3 3"]:
        """Convert this rotation to a 3x3 rotation matrix.

        Returns:
            A shape ``(*, 3, 3)`` orthonormal rotation matrix.
        """
        return six_dimensional_representation_to_matrix(self._representation_6d)

    def as_scipy_rotation(self) -> JaxRotation:
        """Convert this rotation to a JAX scipy ``Rotation`` object.

        Returns:
            A ``jax.scipy.spatial.transform.Rotation`` instance.
        """
        return jax.scipy.spatial.transform.Rotation.from_matrix(self.as_matrix())

    @property
    def shape(self) -> tuple[int, ...]:
        """Batch shape of the rotation."""
        return self._representation_6d.shape[:-1]

    @property
    def inv(self) -> Self:
        """The inverse rotation (transpose of the rotation matrix)."""
        return self.from_matrix(self.as_matrix().swapaxes(-1, -2))  # Transpose = inverse

    def __mul__(self, other: Self) -> Self:
        return self.from_matrix(vmap(lambda a, b: a.as_matrix() @ b.as_matrix())(self, other))

    def __div__(self, other: Self) -> Self:
        return self.from_matrix(vmap(lambda a, b: a.as_matrix() @ b.as_matrix().T)(self, other))

    def __rdiv__(self, other: Self) -> Self:
        return self.from_matrix(vmap(lambda a, b: a.as_matrix().T @ b.as_matrix())(self, other))

    def __getitem__(self, item: slice) -> Self:
        return Rotation(self._representation_6d[item])

    def __str__(self) -> str:
        return f"representation: {jnp.array_str(self._representation_6d)}"


class CoordinateSystem(eqx.Module):
    """A coordinate system defined by a translation and rotation in CXI
    coordinates.

    Represents a rigid-body transformation (rotation + translation) of a local
    coordinate frame relative to the global CXI frame. The CXI convention defines
    z along the incoming beam direction and y vertically.

    The translation can be stored in either the global frame or the local frame,
    controlled by ``model_in_local_frame``. When ``model_in_local_frame=True``,
    the internal translation is stored in the local (rotated) frame and converted
    to global coordinates on access via the :py:attr:`translation` property.

    Attributes:
        rotation: The :py:class:`~ptyrax.spatial.Rotation` of this coordinate system.
        model_in_local_frame: If ``True``, the internal translation is stored in the
            rotated local frame rather than the global frame.

    Example:
        >>> import jax.numpy as jnp
        >>> from ptyrax.spatial import CoordinateSystem, Rotation
        >>> cs = CoordinateSystem(
        ...     rotation=Rotation.from_matrix(jnp.eye(3)),
        ...     translation=jnp.array([1.0, 0.0, 0.0]),
        ... )
        >>> cs.translation
    """

    rotation: Rotation = eqx.field(default_factory=lambda: Rotation.from_matrix(jnp.eye(3)))

    _translation: Float[Array, "* 3"] = eqx.field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))

    model_in_local_frame: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        rotation: Rotation = Rotation.from_matrix(jnp.eye(3)),
        translation: Float[Array, "* 3"] = jnp.array([0.0, 0.0, 0.0]),
        model_in_local_frame: bool = False,
        **kwargs,
    ) -> None:
        if model_in_local_frame:
            # This is called outside of the jit-boundary, so we should resolve parametrizations here
            translation_resolved = resolve_array_parametrizations(translation)
            rotation_resolved = resolve_array_parametrizations(rotation)
            translation_local = jnp.einsum(
                "...ij, ...j -> ...i",
                rotation_resolved.as_matrix(),
                translation_resolved,
            )
            if isinstance(translation, ArrayParametrization):
                # keep the same type of parametrization
                translation_local = type(translation)(translation_local)

            translation = translation_local
        self._translation = translation
        self.rotation = rotation
        self.model_in_local_frame = model_in_local_frame

    @property
    def translation(self) -> Float[Array, "* 3"]:
        """Translation vector in the global CXI frame.

        When ``model_in_local_frame=True``, the stored local-frame translation
        is rotated into the global frame before being returned.
        """
        if not self.model_in_local_frame:
            return self._translation
        internal_translation = self._translation
        # Translate from local to global
        external_frame_translation = jnp.einsum(
            "...ij, ...j -> ...i", self.rotation.inv.as_matrix(), internal_translation
        )
        return external_frame_translation

    @property
    def translation_internal(self) -> Float[Array, "* 3"]:
        """Translation vector in the local (rotated) frame.

        When ``model_in_local_frame=False``, the stored global-frame translation
        is rotated into the local frame before being returned.
        """
        if self.model_in_local_frame:
            return self._translation
        external_translation = self.translation
        # Translate from global to local
        internal_frame_translation = jnp.einsum("...ij, ...j -> ...i", self.rotation.as_matrix(), external_translation)
        return internal_frame_translation

    @property
    def x_axis(self) -> Float[Array, "* 3"]:
        """Unit vector along the local x-axis, expressed in global
        coordinates."""
        return self.rotation.as_matrix()[:, 0]

    @property
    def y_axis(self) -> Float[Array, "* 3"]:
        """Unit vector along the local y-axis, expressed in global
        coordinates."""
        return self.rotation.as_matrix()[:, 1]

    @property
    def z_axis(self) -> Float[Array, "* 3"]:
        """Unit vector along the local z-axis, expressed in global
        coordinates."""
        return self.rotation.as_matrix()[:, 2]

    @property
    def shape(self) -> tuple[int, ...]:
        """Batch shape of the coordinate system."""
        return self._translation.shape[:-1]

    def __getitem__(self, item: slice) -> Self:
        return CoordinateSystem(
            rotation=self.rotation[item],
            translation=self.translation[item],
        )

    def __str__(self) -> str:
        string_representation = """rotation: {jnp.array_str(self.rotation.as_matrix())}
                translation: {jnp.array_str(self.translation)}"""
        return string_representation


class SamplingGrid(eqx.Module):
    """A discrete sampling grid with pixel sizes and optional FFT-shift
    awareness.

    Represents a regular 2D or 3D grid centered at the origin (or shifted by
    ``origin_shift``), with coordinates following CXI convention: x along
    dimension 0, y along dimension 1, using numpy ``'ij'`` indexing.

    Grid coordinates range from ``-N//2`` to ``(N+1)//2 - 1`` along each axis,
    scaled by the corresponding pixel size. When ``fftshifted=True``, the
    coordinate arrays are reordered to match FFT output layout.

    Attributes:
        shape: Grid dimensions as ``(n_x, n_y)`` or ``(n_x, n_y, n_z)``.
        pixel_size: Physical size of each pixel along each dimension, shape ``(*, d)``.
        origin_shift: Optional shift of the grid origin, shape ``(d,)``.
        fftshifted: If ``True``, coordinate arrays are fftshifted.

    Example:
        >>> import jax.numpy as jnp
        >>> from ptyrax.spatial import SamplingGrid
        >>> grid = SamplingGrid(shape=(64, 64), pixel_size=jnp.array([1e-6, 1e-6]))
        >>> grid.x  # 1D coordinates along x
        >>> grid.meshgrid  # Full 2D coordinate meshgrid
    """

    shape: tuple[int, ...] = eqx.field(static=True, converter=lambda x: tuple(x))
    pixel_size: Float[Array, "* d"] | np.ndarray
    origin_shift: Float[Array, " d"] | np.ndarray = eqx.field(default=None)
    fftshifted: bool = eqx.field(static=True, default=False)

    @classmethod
    def from_tuples(
        cls,
        shape: tuple[int | float],
        pixel_size: tuple[float],
        origin_shift: tuple[float] = None,
        fftshifted: bool = False,
    ) -> "SamplingGrid":
        """Construct a :py:class:`~ptyrax.spatial.SamplingGrid` from plain
        tuples.

        Convenience constructor that converts tuple arguments to the appropriate
        array types.

        Args:
            shape: Grid dimensions, e.g. ``(64, 64)``.
            pixel_size: Physical pixel size per dimension, e.g. ``(1e-6, 1e-6)``.
            origin_shift: Optional origin offset per dimension.
            fftshifted: Whether the grid coordinates should be FFT-shifted.

        Returns:
            A new :py:class:`~ptyrax.spatial.SamplingGrid` instance.
        """
        int_cast_shape = tuple(int(el) for el in shape)
        return cls(
            shape=int_cast_shape, pixel_size=jnp.array(pixel_size), origin_shift=origin_shift, fftshifted=fftshifted
        )

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.pixel_size.reshape(-1)):
            raise ValueError(
                f"The sampling grid shape ({len(self.shape)}D) and pixel_size ({len(self.pixel_size.reshape(-1))}D) "
                "must have the same length."
            )
        if len(self.shape) not in (2, 3):
            raise ValueError(f"The sampling grid shape must be 2D or 3D, got {len(self.shape)}D.")

    def __str__(self) -> str:
        return f"shape: {self.shape} \n \
            pixel_size: {self.pixel_size}"

    def __getitem__(self, idx: int) -> int:
        if self.origin_shift is None:
            return SamplingGrid(shape=self.shape, pixel_size=self.pixel_size[idx], origin_shift=None)
        return SamplingGrid(shape=self.shape, pixel_size=self.pixel_size[idx], origin_shift=self.origin_shift[idx])

    @property
    def meshgrid(self) -> Float[Array, "d N_x N_y"] | Float[Array, "d N_x N_y N_z"]:
        """Full coordinate meshgrid with shape ``(d, N_x, N_y)`` or ``(d, N_x,
        N_y, N_z)``."""
        if self.ndim == 2:
            return jnp.stack(jnp.meshgrid(self.x, self.y, indexing="ij"), axis=0)
        return jnp.stack(jnp.meshgrid(self.x, self.y, self.z, indexing="ij"), axis=0)

    @property
    def n_x(self) -> int:
        """Number of grid points along the x-axis."""
        return self.shape[0]

    @property
    def n_x_max(self) -> int:
        """Maximum x index (inclusive) in the centered grid."""
        return (self.n_x + 1) // 2 - 1  # off-by-one so this is actually the largest value in arange

    @property
    def n_x_min(self) -> int:
        """Minimum x index in the centered grid."""
        return -(self.n_x // 2)

    def _n_axis(self, axis: int) -> int:
        """Number of grid points along a given axis."""
        if axis >= len(self.shape):
            raise ValueError(f"The grid is {self.ndim}D, axis {axis} is not defined")
        return self.shape[axis]

    def _n_axis_max(self, axis: int) -> int:
        """Maximum index (inclusive) in the centered grid for a given axis."""
        n = self._n_axis(axis)
        return (n + 1) // 2 - 1

    def _n_axis_min(self, axis: int) -> int:
        """Minimum index in the centered grid for a given axis."""
        n = self._n_axis(axis)
        return -(n // 2)

    def _axis_pixel_size(self, axis: int) -> float:
        """Physical pixel size along a given axis."""
        if axis >= len(self.shape):
            raise ValueError(f"The grid is {self.ndim}D, axis {axis} pixel_size is not defined")
        return self.pixel_size[..., axis]

    def _get_coordinates_1d(self, axis: int) -> Float[Array, " n"]:
        """1D array of physical coordinates along a given axis."""
        coords = jnp.arange(self._n_axis_min(axis), self._n_axis_max(axis) + 1) * self._axis_pixel_size(axis)
        return jnp.fft.fftshift(coords) if self.fftshifted else coords

    @property
    def x_pixel_size(self) -> float:
        """Physical pixel size along the x-axis."""
        return self._axis_pixel_size(0)

    @property
    def x_max(self) -> float:
        """Maximum physical x-coordinate."""
        return self._n_axis_max(0) * self.x_pixel_size

    @property
    def x_min(self) -> float:
        """Minimum physical x-coordinate."""
        return self._n_axis_min(0) * self.x_pixel_size

    @property
    def x(self) -> Float[Array, "n_x"]:
        """1D array of physical x-coordinates."""
        return self._get_coordinates_1d(0)

    @property
    def n_y(self) -> int:
        """Number of grid points along the y-axis."""
        return self.shape[1]

    @property
    def n_y_max(self) -> int:
        """Maximum y index (inclusive) in the centered grid."""
        return self._n_axis_max(1)

    @property
    def n_y_min(self) -> int:
        """Minimum y index in the centered grid."""
        return self._n_axis_min(1)

    @property
    def y_pixel_size(self) -> float:
        """Physical pixel size along the y-axis."""
        return self._axis_pixel_size(1)

    @property
    def y_max(self) -> float:
        """Maximum physical y-coordinate."""
        return self._n_axis_max(1) * self.y_pixel_size

    @property
    def y_min(self) -> float:
        """Minimum physical y-coordinate."""
        return self._n_axis_min(1) * self.y_pixel_size

    @property
    def y(self) -> Float[Array, "n_y"]:
        """1D array of physical y-coordinates."""
        return self._get_coordinates_1d(1)

    @property
    def n_z(self) -> int:
        """Number of grid points along the z-axis.

        Raises:
            ValueError: If the grid is 2D.
        """
        if self.ndim == 2:
            raise ValueError("The grid is 2D, N_z is not defined")
        return self.shape[2]

    @property
    def n_z_max(self) -> int:
        """Maximum z index (inclusive) in the centered grid."""
        return self._n_axis_max(2)

    @property
    def n_z_min(self) -> int:
        """Minimum z index in the centered grid."""
        return self._n_axis_min(2)

    @property
    def z_pixel_size(self) -> float:
        """Physical pixel size along the z-axis.

        Raises:
            ValueError: If the grid is 2D.
        """
        return self._axis_pixel_size(2)

    @property
    def z_max(self) -> float:
        """Maximum physical z-coordinate."""
        return self.n_z_max * self.z_pixel_size

    @property
    def z_min(self) -> float:
        """Minimum physical z-coordinate."""
        return self.n_z_min * self.z_pixel_size

    @property
    def z(self) -> Float[Array, "n_z"]:
        """1D array of physical z-coordinates."""
        return self._get_coordinates_1d(2)

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (2 or 3)."""
        return len(self.shape)

    @property
    def xx(self) -> Float[Array, "N_x N_y"] | Float[Array, "N_x N_y N_z"]:
        """X-component of the coordinate meshgrid."""
        return self.meshgrid[0]

    @property
    def yy(self) -> Float[Array, "N_x N_y"] | Float[Array, "N_x N_y N_z"]:
        """Y-component of the coordinate meshgrid."""
        return self.meshgrid[1]

    @property
    def zz(self) -> Float[Array, "N_x N_y N_z"] | Float[Array, "N_x N_y N_z"]:
        """Z-component of the coordinate meshgrid (3D grids only)."""
        return self.meshgrid[2]

    @property
    def rr(self) -> Float[Array, "N_x N_y"] | Float[Array, "N_x N_y N_z"]:
        """Radial distance from the origin at each grid point."""
        if self.ndim == 2:
            return jnp.sqrt(self.xx**2 + self.yy**2)
        else:
            return jnp.sqrt(self.xx**2 + self.yy**2 + self.zz**2)

    @property
    def rr2(self) -> Float[Array, "N_x N_y"] | Float[Array, "N_x N_y N_z"]:
        """Squared radial distance from the origin at each grid point."""
        if self.ndim == 2:
            return self.xx**2 + self.yy**2
        else:
            return self.xx**2 + self.yy**2 + self.zz**2

    @property
    def x_bounds(self) -> Float[Array, "... 2"]:
        """Min and max physical x-coordinates as a ``(2,)`` array."""
        return jnp.stack([self.x_min, self.x_max], axis=-1)

    @property
    def y_bounds(self) -> Float[Array, "... 2"]:
        """Min and max physical y-coordinates as a ``(2,)`` array."""
        return jnp.stack([self.y_min, self.y_max], axis=-1)

    @property
    def z_bounds(self) -> Float[Array, "... 2"]:
        """Min and max physical z-coordinates as a ``(2,)`` array."""
        return jnp.stack([self.z_min, self.z_max], axis=-1)

    @property
    def bounds(self) -> Float[Array, "... d 2"]:
        """Coordinate bounds as a ``(*, d, 2)`` array of ``[min, max]`` per
        axis.

        If ``origin_shift`` is set, it is added to the bounds.
        """
        if self.ndim == 2:
            bounds = jnp.stack([self.x_bounds, self.y_bounds], axis=-2)
        else:
            bounds = jnp.stack([self.x_bounds, self.y_bounds, self.z_bounds], axis=-2)
        if self.origin_shift is not None:
            bounds = bounds + self.origin_shift[:, np.newaxis]
        return bounds

    @property
    def field_of_view(self) -> Float[Array, "d 2"]:
        """Alias for :py:attr:`bounds`."""
        return self.bounds

    @property
    def extent(self) -> Float[Array, "... 2d"]:
        """Flattened bounds as a ``(*, 2d)`` array, e.g. ``(x_min, x_max,
        y_min, y_max)``."""
        if self.ndim == 2:
            return self.bounds.reshape(self.bounds.shape[:-2] + (4,))
        else:
            return self.bounds.reshape(self.bounds.shape[:-2] + (6,))

    def edges(self, n_per_edge: int = 100) -> Float[Array, "N 3"]:
        """Generate 3D coordinates along the four edges of the 2D grid
        boundary.

        Returns points tracing the rectangular boundary of the grid in the
        local x-y plane (z=0), useful for visualization.

        Args:
            n_per_edge: Number of sample points along each edge.

        Returns:
            An ``(N, 3)`` array of boundary coordinates with z=0.
        """
        x_edge = jnp.concatenate(
            [
                jnp.tile(jnp.array([self.x_min]), n_per_edge),  # Constant x as a function of y
                jnp.tile(jnp.array([self.x_max]), n_per_edge),
                jnp.linspace(self.x_min, self.x_max, n_per_edge),
                jnp.linspace(self.x_min, self.x_max, n_per_edge),
                jnp.array([0.0]),
            ]
        )
        y_edge = jnp.concatenate(
            [
                jnp.linspace(self.y_min, self.y_max, n_per_edge),
                jnp.linspace(self.y_min, self.y_max, n_per_edge),
                jnp.tile(jnp.array([self.y_min]), n_per_edge),
                jnp.tile(jnp.array([self.y_max]), n_per_edge),
                jnp.array([0.0]),
            ]
        )
        z_edge = jnp.zeros_like(x_edge)

        edges = jnp.stack([x_edge, y_edge, z_edge], axis=-1)
        return edges

    def subgrid_coordinates(
        self,
        n_x: int = 20,
        n_y: int = 20,
    ) -> Float[Array, "n_x n_y 3"]:
        """Generate a coarse subgrid of 3D coordinates spanning the grid
        extent.

        Useful for visualization or sampling a sparse set of points across the grid.

        Args:
            n_x: Number of points along the x-axis.
            n_y: Number of points along the y-axis.

        Returns:
            An ``(n_x * n_y, 3)`` array of coordinates with z=0.
        """
        x = jnp.linspace(self.x_min, self.x_max, n_x, endpoint=True)
        y = jnp.linspace(self.y_min, self.y_max, n_y, endpoint=True)
        xx, yy = jnp.meshgrid(x, y, indexing="ij")
        xx, yy = (xx.flatten(), yy.flatten())
        return jnp.stack([xx, yy, jnp.zeros_like(xx)], axis=-1)

    def lines_for_plot(self, n_x: int = 20, n_y: int = 20) -> Float[Array, "N 2"]:
        """Generate grid lines for 2D plotting.

        Produces horizontal and vertical line segments spanning the grid, suitable
        for overlaying on a plot.

        Args:
            n_x: Number of vertical grid lines.
            n_y: Number of horizontal grid lines.

        Returns:
            An array of line segments, each of shape ``(2, N)``.
        """
        x = jnp.linspace(self.x_min, self.x_max, n_x, endpoint=True)
        y = jnp.linspace(self.y_min, self.y_max, n_y, endpoint=True)
        lines = []
        lines.extend(jnp.stack([jnp.full_like(y, x_i), y], axis=-1) for x_i in x)
        lines.extend(jnp.stack([x, jnp.full_like(x, y_i)], axis=-1) for y_i in y)
        return jnp.asarray(lines)

    def to_far_field(
        self,
        wavelength: Float[Array, ""] | float = 1.0,
        propagation_distance: Float[Array, ""] | float = 1.0,
        fftshifted: bool = False,
    ) -> Self:
        r"""Compute the reciprocal-space (far-field) grid corresponding to this
        grid.

        Applies the Fraunhofer diffraction relation to convert pixel sizes from
        real space to reciprocal space:

        $$\Delta q_i = \frac{\lambda \cdot z}{\Delta x_i \cdot N_i}$$

        where $\lambda$ is the wavelength, $z$ is the propagation distance,
        $\Delta x_i$ is the real-space pixel size, and $N_i$ is the number of
        pixels along axis $i$.

        Args:
            wavelength: Radiation wavelength (scalar).
            propagation_distance: Propagation distance to the far-field plane (scalar).
            fftshifted: Whether the returned grid should be FFT-shifted.

        Returns:
            A new :py:class:`~ptyrax.spatial.SamplingGrid` with reciprocal-space pixel sizes.

        Raises:
            ValueError: If ``wavelength`` is not scalar.
        """
        if not isinstance(wavelength, float) and wavelength.shape != ():
            raise ValueError(
                "Wavelength was not scalar in computing far-field. If multiple wavelengths are "
                "used, these must be vmapped over"
            )
        # explicit indices must be used to avoid casting to either jnp.array or np.array
        new_pixel_size = tuple(
            wavelength * propagation_distance / (px * shp) for px, shp in zip(self.pixel_size, self.shape)
        )

        if isinstance(self.pixel_size, jax.Array):
            new_pixel_size = jnp.asarray(new_pixel_size)
        if isinstance(self.pixel_size, np.ndarray):
            new_pixel_size = np.asarray(new_pixel_size)

        return SamplingGrid.from_tuples(shape=self.shape, pixel_size=new_pixel_size, fftshifted=fftshifted)

    def __plot__(
        self,
        fig: Figure = None,
        gs: SubplotSpec = None,
        show: bool = True,
        **kwargs,
    ) -> tuple[Figure, SubplotSpec, list[Line2D]]:
        if gs is None:
            fig = plt.figure(1, 1)
            gs = fig.add_gridspec(1, 1)[0]

        ax = fig.add_subplot(gs[0, 0])
        grid = self.meshgrid.reshape(2, -1)
        line = ax.plot(*grid, ".", **kwargs)
        if show:
            plt.show()
        return fig, gs, line


class YOnlyRotation(Rotation):
    """A rotation constrained to rotate only about the y-axis.

    Useful for samples in simple reflection geometries where only a rotation
    around the vertical (y) axis are presumed to be present. The y-axis component
    of the 6D representation is fixed to ``[0, 1, 0]``.

    Args:
        representation_initializer: A callable returning a shape ``(3,)`` array
            representing the x-axis direction of the rotation.
    """

    representation = Float[Array, "3"]

    def __init__(self, representation_initializer: Callable[[], Float[Array, "3"]]) -> Self:
        self.representation = representation_initializer()

    @property
    def representation_6d(self) -> Float[Array, "6"]:
        y_axis_part = jnp.tile(jnp.array([0.0, 1.0, 0.0]), (self.representation.shape[:-1], 1))
        return jnp.concatenate([self.representation, y_axis_part], axis=-1)


def meshgrid(
    shape: tuple[int, int],
    pixel_size: Float[Array, "2"] = np.array([1.0, 1.0]),
) -> Float[Array, "2 n m"]:
    """Create a 2D coordinate meshgrid centered at the origin.

    Generates coordinate arrays spanning ``[-N/2, N/2)`` along each axis,
    scaled by the corresponding pixel size.

    Args:
        shape: Grid dimensions as ``(n_rows, n_cols)``.
        pixel_size: Physical pixel size as ``(dy, dx)``.

    Returns:
        A ``(2, n, m)`` array where ``[0]`` is the x-coordinates and ``[1]``
        is the y-coordinates.
    """
    y = jnp.arange(-shape[0] / 2, shape[0] / 2) * pixel_size[0]
    x = jnp.arange(-shape[1] / 2, shape[1] / 2) * pixel_size[1]
    return jnp.stack(jnp.meshgrid(x, y), axis=0)


def convert_coordinates_into_indices(
    coordinates: Float[Array, "... 2"],
    target_sampling: SamplingGrid,
) -> tuple[Float[Array, "... 2"], Bool[Array, "..."]]:
    """Map physical coordinates to array indices on a target grid.

    Converts physical (x, y) coordinates into floating-point array indices
    relative to the target :py:class:`~ptyrax.spatial.SamplingGrid`. Also
    returns a boolean mask indicating which coordinates fall within the grid
    boundaries.

    Args:
        coordinates: Physical coordinates with shape ``(..., 2)`` where the
            leading dimension indexes ``(x, y)``.
        target_sampling: The target grid defining pixel sizes and shape.

    Returns:
        A tuple of:
            - ``indices``: Floating-point array indices, shape ``(2, ...)``.
            - ``mask``: Boolean array indicating in-bounds coordinates.
    """
    # pixel_size = jnp.flip(target_sampling.pixel_size)  # x -> j,  y -> i
    # coordinates = jnp.flip(coordinates, axis=0)  # x -> j,  y -> i
    pixel_number = jnp.array(target_sampling.shape)
    n_dim = len(coordinates.shape) - 1
    # Dimensionality of coordinates is the leading dimension
    extra_indices = (jnp.newaxis,) * n_dim

    indices = coordinates / target_sampling.pixel_size[:, *extra_indices] + pixel_number[:, *extra_indices] / 2

    # flip the y-axis: (0,0) is at the top-left corner
    # indices = jnp.transpose(indices, (0, -1, -2))
    indices = jnp.stack((indices[0], indices[1]), axis=0)
    mask = jnp.logical_and(indices > 0, indices < pixel_number[:, *extra_indices])
    mask = jnp.logical_and(*mask)
    return indices, mask


@jax.custom_jvp
def angle_safe(z: Complex[Array, "..."], default: float = 0.0, eps: float = 1e-12) -> Float[Array, "..."]:
    """Angle with a fixed value at z==0 and safe gradients.

    Args:
      z: complex array
      default: scalar angle to return at z==0
      eps: threshold to treat |z|^2 as zero (avoid tiny denominators)
    """
    # primal: choose default when magnitude is (near) zero
    mask = jnp.abs(z) ** 2 <= eps
    return jnp.where(mask, default, jnp.angle(z))


def _sliced_rounded(
    x: Float[Array, "... w0 h0"], target_center_idx: Float[Array, "2"], target_shape: tuple[int, int]
) -> Float[Array, "... w1 h1"]:
    "Shift by slicing (no interpolation). Pixel size should be equal."
    (*_, w0, h0) = x.shape
    (w1, h1) = target_shape
    start_x = jnp.round(target_center_idx[0] + w0 / 2 - w1 / 2).astype(jnp.int32)
    start_y = jnp.round(target_center_idx[1] + h0 / 2 - h1 / 2).astype(jnp.int32)

    sliced_x = jax.lax.dynamic_slice(
        x,
        start_indices=(0,) * (x.ndim - 2) + (start_x, start_y),
        slice_sizes=x.shape[:-2] + (w1, h1),
    )
    return sliced_x


def _sliced_interpolated(
    x: Float[Array, "... w0 h0"],
    target_center_idx: Float[Array, "2"],
    target_shape: tuple[int, int],
    mode: Literal["real_imaginary", "amplitude_phase"] = "real_imaginary",
) -> Float[Array, "... w1 h1"]:
    (*_, w0, h0) = x.shape
    (w1, h1) = target_shape
    start_x = target_center_idx[0] + w0 / 2 - w1 / 2
    start_y = target_center_idx[1] + h0 / 2 - h1 / 2

    left_x = jnp.floor(start_x).astype(jnp.int32)
    top_y = jnp.floor(start_y).astype(jnp.int32)
    right_x = left_x + 1
    bottom_y = top_y + 1

    sliced_top_left = jax.lax.dynamic_slice(
        x,
        start_indices=(0,) * (x.ndim - 2) + (left_x, top_y),
        slice_sizes=x.shape[:-2] + (w1, h1),
    )
    sliced_top_right = jax.lax.dynamic_slice(
        x,
        start_indices=(0,) * (x.ndim - 2) + (right_x, top_y),
        slice_sizes=x.shape[:-2] + (w1, h1),
    )
    sliced_bottom_left = jax.lax.dynamic_slice(
        x,
        start_indices=(0,) * (x.ndim - 2) + (left_x, bottom_y),
        slice_sizes=x.shape[:-2] + (w1, h1),
    )
    sliced_bottom_right = jax.lax.dynamic_slice(
        x,
        start_indices=(0,) * (x.ndim - 2) + (right_x, bottom_y),
        slice_sizes=x.shape[:-2] + (w1, h1),
    )
    alpha_x = start_x - left_x
    alpha_y = start_y - top_y

    def _interp(corners):  # noqa: ANN001
        (top_left, top_right, bottom_left, bottom_right) = corners
        top = (1 - alpha_x) * top_left + alpha_x * top_right
        bottom = (1 - alpha_x) * bottom_left + alpha_x * bottom_right
        interpolated = (1 - alpha_y) * top + alpha_y * bottom
        return interpolated

    if mode == "real_imaginary":
        interpolated = _interp((sliced_top_left, sliced_top_right, sliced_bottom_left, sliced_bottom_right))
    elif mode == "amplitude_phase":
        interpolated_amplitude = _interp(
            (
                jnp.abs(sliced_top_left),
                jnp.abs(sliced_top_right),
                jnp.abs(sliced_bottom_left),
                jnp.abs(sliced_bottom_right),
            )
        )
        interpolated_total = _interp(
            (
                sliced_top_left,
                sliced_top_right,
                sliced_bottom_left,
                sliced_bottom_right,
            )
        )
        interpolated = interpolated_amplitude * phase_only_exp(angle_safe(interpolated_total))
    return interpolated


def shift_with_interpolation_equal_pixel_size(
    x: Float[Array, "... w0 h0"],
    target_center_idx: Float[Array, "2"],
    target_shape: tuple[int, int],
    interpolation_mode: Literal["rounded", "real_imaginary", "amplitude_phase"] = "rounded",
) -> Float[Array, "... w1 h1"]:
    """Extract a shifted sub-region from an array when source and target pixel
    sizes match.

    Crops a region of size ``target_shape`` from ``x``, centered at
    ``target_center_idx`` (in pixel units relative to the center of ``x``).
    The shift can be rounded to the nearest pixel or interpolated.

    Args:
        x: Source array of shape ``(..., w0, h0)``. If 3D, the leading
            dimension is treated as independent channels.
        target_center_idx: Sub-pixel center offset as ``(dx, dy)`` in index space.
        target_shape: Output spatial dimensions ``(w1, h1)``.
        interpolation_mode: One of ``'rounded'`` (nearest-pixel slice),
            ``'real_imaginary'`` (bilinear on real/imag parts), or
            ``'amplitude_phase'`` (bilinear on amplitude, phase-aware).

    Returns:
        The shifted sub-region with shape ``(..., w1, h1)``.
    """

    def interpolate_single_channel(channel: Float[Array, "..."]) -> Float[Array, "..."]:
        if interpolation_mode == "rounded":
            interpolated_output = _sliced_rounded(channel, target_center_idx, target_shape)
        elif interpolation_mode in ("real_imaginary", "amplitude_phase"):
            interpolated_output = _sliced_interpolated(channel, target_center_idx, target_shape)
        else:
            raise ValueError(
                "interpolation mode for equal pixels must be one of ['rounded', 'real_imaginary', 'amplitude_phase']. "
                f"Got {interpolation_mode} instead"
            )
        return interpolated_output

    if x.ndim == 3:
        interpolator = jax.vmap(interpolate_single_channel)
    else:
        interpolator = interpolate_single_channel

    return interpolator(x)


@gin.configurable()
def shift_with_interpolation_unequal_pixel_size(
    x: Float[Array, "... w0 h0"],
    original_pixel_size: Float[Array, "2"] | tuple[float, float],
    target_center_idx: Float[Array, "2"],
    target_shape: tuple[int, int],
    target_pixel_size: Float[Array, "2"] | tuple[float, float],
    interpolation_mode: Literal["real_imaginary", "amplitude_phase"] = "real_imaginary",
) -> Float[Array, "... w1 h1"]:
    """Extract a shifted and resampled sub-region when pixel sizes differ.

    Interpolates values from ``x`` onto a target grid that may have a different
    pixel size, using ``jax.scipy.ndimage.map_coordinates`` for the resampling.

    Args:
        x: Source array of shape ``(..., w0, h0)``.
        original_pixel_size: Pixel size of the source array as ``(px_x, px_y)``.
        target_center_idx: Center offset in source-pixel units as ``(dx, dy)``.
        target_shape: Output spatial dimensions ``(w1, h1)``.
        target_pixel_size: Pixel size of the target grid as ``(px_x, px_y)``.
        interpolation_mode: Either ``'real_imaginary'`` (bilinear on real/imag)
            or ``'amplitude_phase'`` (bilinear on amplitude, phase-aware).

    Returns:
        The resampled sub-region with shape ``(..., w1, h1)``.
    """
    (*_, w0, h0) = x.shape
    (w1, h1) = target_shape

    # relative_pixel_size = [t/o for t, o in zip(target_pixel_size, original_pixel_size)]

    # grid_x = jnp.arange(w)  * relative_pixel_size[0]
    # grid_y = jnp.arange(h) * relative_pixel_size[1]

    # Compute index offsets for target grid, centered around its own center
    grid_x = (jnp.arange(w1) - (w1) / 2) * (target_pixel_size[0] / original_pixel_size[0])
    grid_y = (jnp.arange(h1) - (h1) / 2) * (target_pixel_size[1] / original_pixel_size[1])

    grid_x, grid_y = jnp.meshgrid(grid_x, grid_y, indexing="ij")
    # Rotating about target_center
    # TODO include rotation of the indexing grid
    # rotated_grid = jnp.einsum('ij, mnj -> mni', target_rotation_matrix,
    # jnp.stack([grid_x, grid_y, jnp.zeros_like(grid_x)], axis=-1))

    # Translate to source index space: center of source + center offset
    grid_x = grid_x + target_center_idx[0] + w0 / 2
    grid_y = grid_y + target_center_idx[1] + h0 / 2

    grid = jnp.stack([grid_x, grid_y], axis=0)

    def interpolate_single_channel(channel: Float[Array, "..."]) -> Float[Array, "..."]:
        if interpolation_mode == "real_imaginary":
            interpolated_output = map_coordinates(channel, grid, order=1, mode="constant")
        elif interpolation_mode == "amplitude_phase":
            interpolated_amplitude = map_coordinates(jnp.abs(channel), grid, order=1, mode="constant")
            interpolated_total = map_coordinates(channel, grid, order=1, mode="constant")
            interpolated_output = interpolated_amplitude * phase_only_exp(angle_safe(interpolated_total))
        else:
            raise ValueError(
                "interpolation mode for unequal pixels must be one of ['real_imaginary', 'amplitude_phase']."
                f"Got {interpolation_mode}"
            )
        return interpolated_output

    if x.ndim == 3:
        interpolator = jax.vmap(interpolate_single_channel)
    else:
        interpolator = interpolate_single_channel

    return interpolator(x)


def interpolate_grid_to_grid(
    samples: Complex[Array, "* m0 n0"],
    source_grid: SamplingGrid = None,
    source_coordinate_system: CoordinateSystem = CoordinateSystem(),
    target_grid: SamplingGrid = None,
    target_coordinate_system: CoordinateSystem = CoordinateSystem(),
    interpolation_mode: Literal["real_imaginary", "amplitude_phase", "sliced_rounded", "sliced_interpolated"] = None,
    equal_pixel_size: bool = False,
) -> Complex[Array, "* m1 n1"]:
    """Interpolate complex-valued samples from a source grid onto a target
    grid.

    Handles both equal and unequal pixel sizes between source and target grids,
    accounting for the relative translation between their coordinate systems.
    The translation is projected into the source's local frame to determine the
    pixel-space offset.

    Args:
        samples: Complex array of shape ``(*, m0, n0)`` on the source grid.
        source_grid: The :py:class:`~ptyrax.spatial.SamplingGrid` of the source data.
        source_coordinate_system: The :py:class:`~ptyrax.spatial.CoordinateSystem`
            of the source grid.
        target_grid: The :py:class:`~ptyrax.spatial.SamplingGrid` to interpolate onto.
            Defaults to ``source_grid`` if not provided.
        target_coordinate_system: The :py:class:`~ptyrax.spatial.CoordinateSystem`
            of the target grid.
        interpolation_mode: Interpolation strategy. See
            :py:func:`~ptyrax.spatial.shift_with_interpolation_equal_pixel_size` and
            :py:func:`~ptyrax.spatial.shift_with_interpolation_unequal_pixel_size`.
        equal_pixel_size: If ``True``, use the faster equal-pixel-size path.

    Returns:
        The interpolated samples on the target grid, shape ``(*, m1, n1)``.

    Raises:
        ValueError: If ``source_grid`` is not provided.
    """
    if source_grid is None:
        raise ValueError("Source_grid must be provided in call to interpolate_grid_to_grid")
    if target_grid is None:
        target_grid = source_grid
        equal_pixel_size = True
    kwargs = {"interpolation_mode": interpolation_mode} if interpolation_mode is not None else {}

    projected_translation = source_coordinate_system.translation_internal
    source_pixel_size = jnp.array((source_grid.x_pixel_size, source_grid.y_pixel_size))
    projected_translation_idx = projected_translation[:2] / source_pixel_size
    if equal_pixel_size:
        interpolated_samples = shift_with_interpolation_equal_pixel_size(
            samples, projected_translation_idx, target_grid.shape, **kwargs
        )
        return interpolated_samples
    else:
        target_pixel_size = jnp.array((target_grid.x_pixel_size, target_grid.y_pixel_size))
        interpolated_samples = shift_with_interpolation_unequal_pixel_size(
            samples,
            source_pixel_size,
            projected_translation_idx,
            target_grid.shape,
            target_pixel_size,
            **kwargs,
        )
    return interpolated_samples


def draw_axis_arrow(
    axis_label: str,
    xy_start: tuple[float, float],
    xy_end: tuple[float, float],
    ax: Axes,
    **kwargs,
) -> Axes:
    """Draw a labeled arrow on a matplotlib axes to indicate a coordinate axis.

    Args:
        axis_label: Text label for the axis (e.g. ``'$x_s$'``).
        xy_start: ``(x, y)`` coordinates of the arrow tip.
        xy_end: ``(x, y)`` coordinates of the arrow tail.
        ax: The matplotlib axes to draw on.
        **kwargs: Additional keyword arguments passed to ``ax.annotate``.

    Returns:
        The axes with the arrow drawn.
    """
    ax.annotate(
        "",
        xy=xy_start,
        xytext=xy_end,
        arrowprops=kwargs.pop("arrowprops", dict(arrowstyle="->", color=kwargs.get("color", "black"))),
        **kwargs,
    )
    ax.annotate(
        axis_label,
        xy=xy_start,
        xytext=(5, 5),
        textcoords="offset points",
        color=kwargs.get("color", "black"),
    )
    return ax


def plot_geometry(
    sample_coordinates: CoordinateSystem,
    detector_coordinates: CoordinateSystem,
    detector_coordinate_edges: ArrayLike,
    arrow_length_ratio: float = 0.1,
    fig: Figure = None,
    gs: SubplotSpec = None,
) -> tuple[Figure, Axes]:
    """Plot the experimental geometry in the x-z plane.

    Visualizes the sample and detector positions along with their local
    coordinate axes (x in red, z in blue) projected onto the x-z scattering
    plane.

    Args:
        sample_coordinates: The :py:class:`~ptyrax.spatial.CoordinateSystem`
            describing sample positions and orientations.
        detector_coordinates: The :py:class:`~ptyrax.spatial.CoordinateSystem`
            describing detector position and orientation.
        detector_coordinate_edges: Array of detector boundary coordinates used
            for plotting detector extent.
        arrow_length_ratio: Length of axis arrows as a fraction of the plot scale.
        fig: Optional existing matplotlib figure.
        gs: Optional matplotlib gridspec subplot.

    Returns:
        A tuple of the matplotlib ``(Figure, Axes)``.
    """
    fig, ax = make_or_reuse_axes(fig, gs)
    ax.plot(detector_coordinate_edges[0, :, 0], detector_coordinate_edges[0, :, 2], ".")
    ax.plot(
        sample_coordinates.translation[:, 0] * 100,
        sample_coordinates.translation[:, 2] * 100,
        ".",
    )
    sample_x = sample_coordinates.rotation.as_matrix()[0].T @ jnp.array([1, 0, 0])
    sample_z = sample_coordinates.rotation.as_matrix()[0].T @ jnp.array([0, 0, 1])
    scale = max((ax.get_xlim()[1] - ax.get_xlim()[0]), (ax.get_ylim()[1] - ax.get_ylim()[0]))
    arrow_length = scale * arrow_length_ratio
    ax.set_aspect("equal", adjustable="box")
    ax = draw_axis_arrow(
        "$x_s$",
        (arrow_length * sample_x[0], arrow_length * sample_x[2]),
        (0, 0),
        ax,
        color="red",
    )
    ax = draw_axis_arrow(
        "$z_s$",
        (arrow_length * sample_z[0], arrow_length * sample_z[2]),
        (0, 0),
        ax,
        color="blue",
    )
    detector_x = detector_coordinates.rotation.as_matrix()[0].T @ jnp.array([1, 0, 0])
    detector_z = detector_coordinates.rotation.as_matrix()[0].T @ jnp.array([0, 0, 1])
    x_d, y_d, z_d = detector_coordinates.translation[0]
    ax = draw_axis_arrow(
        "$x_d$",
        (x_d + arrow_length * detector_x[0], z_d + arrow_length * detector_x[2]),
        (x_d, z_d),
        ax,
        color="red",
    )
    ax = draw_axis_arrow(
        "$z_d$",
        (x_d + arrow_length * detector_z[0], z_d + arrow_length * detector_z[2]),
        (x_d, z_d),
        ax,
        color="blue",
    )

    ax.axis("equal")
    ax.set_xlabel("x")
    ax.set_xlim(np.flip(ax.get_xlim()))
    ax.set_ylabel("z")
    return fig, ax


def R_x(angle: float, **kwargs) -> Float[Array, "3 3"]:  # noqa: N802
    """Construct a 3x3 rotation matrix for rotation about the x-axis.

    Args:
        angle: Rotation angle in degrees.

    Returns:
        A ``(3, 3)`` rotation matrix.
    """
    return JaxRotation.from_rotvec([angle, 0, 0], degrees=True).as_matrix()


def R_y(angle: float, **kwargs) -> Float[Array, "3 3"]:  # noqa: N802
    """Construct a 3x3 rotation matrix for rotation about the y-axis.

    Args:
        angle: Rotation angle in degrees.

    Returns:
        A ``(3, 3)`` rotation matrix.
    """
    return JaxRotation.from_rotvec([0, angle, 0], degrees=True).as_matrix()


def R_z(angle: float, **kwargs) -> Float[Array, "3 3"]:  # noqa: N802
    """Construct a 3x3 rotation matrix for rotation about the z-axis.

    Args:
        angle: Rotation angle in degrees.

    Returns:
        A ``(3, 3)`` rotation matrix.
    """
    return JaxRotation.from_rotvec([0, 0, angle], degrees=True).as_matrix()


@gin.configurable("interpolated_shift")
def shift_with_interpolation(
    x: Float[Array, "..."],
    center: Float[Array, "2"],
    target_shape: tuple[int, int],
) -> Array:
    """Shift and crop an array using bilinear interpolation.

    Extracts a region of ``target_shape`` from ``x`` centered at ``center``
    (in pixel coordinates relative to the array center), using
    ``jax.scipy.ndimage.map_coordinates`` with nearest-neighbor boundary
    handling. The leading dimension of ``x`` is vmapped over as channels.

    Args:
        x: Source array of shape ``(C, w, h)`` where ``C`` is the number of
            channels.
        center: Sub-pixel center offset as ``(cx, cy)`` in index space.
        target_shape: Output spatial dimensions ``(w_out, h_out)``.

    Returns:
        The interpolated sub-region with shape ``(C, w_out, h_out)``.
    """
    (_, w, h) = x.shape
    corner = center - jnp.array(target_shape) / 2
    coords = jnp.array((w / 2 + corner[0], h / 2 + corner[1]))

    # Create a grid of coordinates for the target shape
    grid_x, grid_y = jnp.meshgrid(jnp.arange(target_shape[0]), jnp.arange(target_shape[1]), indexing="ij")
    grid_x = grid_x + coords[0]
    grid_y = grid_y + coords[1]

    grid = jnp.stack([grid_x, grid_y], axis=0)

    def interpolate_single_channel(channel: Float[Array, "..."]) -> Float[Array, "..."]:
        return map_coordinates(channel, grid, order=1, mode="nearest")

    interpolated_values = jax.vmap(interpolate_single_channel)(x)

    return interpolated_values


@angle_safe.defjvp
def angle_safe_jvp(
    primals: Complex[Array, "..."], tangents: Complex[Array, "..."]
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    (z, default, eps), (z_dot, _, _) = primals, tangents

    # primal (same as forward)
    r2 = jnp.abs(z) ** 2
    mask = r2 <= eps
    primal_out = jnp.where(mask, default, jnp.angle(z))

    # SAFE denominator: never zero (or under eps) so division is finite.
    safe_r2 = jnp.maximum(r2, eps)

    # numerator for tangent: Im(conj(z) * dz)
    # note: this is real-valued
    numerator = jnp.imag(jnp.conj(z) * z_dot)

    # compute tangent using safe denominator then zero it where |z|^2 <= eps
    tentative_tangent = numerator / safe_r2
    tangent_out = jnp.where(mask, 0.0, tentative_tangent)

    return primal_out, tangent_out


def rotation_matrix_from_angles(angles: Float[Array, "3 ..."]) -> Float[Array, "... 3 3"]:
    r"""Build a rotation matrix from intrinsic Euler angles (x-y-z order).

    Constructs the combined rotation matrix as $R = R_z \cdot R_y \cdot R_x$
    from three rotation angles given in degrees.

    Args:
        angles: A shape ``(3, ...)`` array of ``(theta_x, theta_y, theta_z)``
            Euler angles in degrees.

    Returns:
        A ``(..., 3, 3)`` rotation matrix.
    """
    theta_x, theta_y, theta_z = jnp.deg2rad(angles)

    r_x = jnp.array([[1, 0, 0], [0, jnp.cos(theta_x), -jnp.sin(theta_x)], [0, jnp.sin(theta_x), jnp.cos(theta_x)]])

    r_y = jnp.array([[jnp.cos(theta_y), 0, jnp.sin(theta_y)], [0, 1, 0], [-jnp.sin(theta_y), 0, jnp.cos(theta_y)]])

    r_z = jnp.array([[jnp.cos(theta_z), -jnp.sin(theta_z), 0], [jnp.sin(theta_z), jnp.cos(theta_z), 0], [0, 0, 1]])

    return r_z @ r_y @ r_x


def phase_only_exp(phase: Float[Array, "... m n"]) -> Complex[Array, "... m n"]:
    r"""Compute a unit-magnitude complex exponential $e^{i\phi}$ using JAX.

    Equivalent to ``jnp.exp(1j * phase)`` but avoids constructing an
    intermediate complex array.

    Args:
        phase: Real-valued phase array in radians.

    Returns:
        Complex array with unit magnitude and the given phase.

    Raises:
        ValueError: If ``phase`` is complex-valued.
    """
    if jnp.iscomplexobj(phase):
        raise ValueError("Input to phase_only_exp should be real-valued phase array.")
    return jnp.cos(phase) + 1j * jnp.sin(phase)


def phase_only_exp_np(phase: Float[Array, "... m n"]) -> Complex[Array, "... m n"]:
    r"""Compute a unit-magnitude complex exponential $e^{i\phi}$ using NumPy.

    NumPy equivalent of :py:func:`~ptyrax.spatial.phase_only_exp`, useful
    outside of JAX-traced contexts.

    Args:
        phase: Real-valued phase array in radians.

    Returns:
        Complex array with unit magnitude and the given phase.

    Raises:
        ValueError: If ``phase`` is complex-valued.
    """
    if np.iscomplexobj(phase):
        raise ValueError("Input to phase_only_exp should be real-valued phase array.")
    return np.cos(phase) + 1j * np.sin(phase)


def reflect(a: Shaped[Array, "... d"], n: Shaped[Array, " d"]) -> Shaped[Array, "... d"]:
    r"""Reflect a vector ``a`` across a plane with unit normal ``n``.

    Computes the reflection $a - 2(a \cdot n)n$.

    Args:
        a: Vector(s) to reflect, shape ``(..., d)``.
        n: Unit normal of the reflection plane, shape ``(d,)``.

    Returns:
        The reflected vector(s), same shape as ``a``.
    """
    return a - 2 * jnp.dot(a, n) * n


def is_orthogonal(a: ArrayLike, b: ArrayLike) -> bool:
    """Check whether two vectors are orthogonal (dot product close to zero)."""
    return jnp.isclose(a @ b, 0.0)


def are_mutually_orthogonal(*args: tuple[ArrayLike, "..."]) -> bool:
    """Check whether all given vectors are mutually orthogonal.

    Args:
        *args: Two or more vectors to test.

    Returns:
        ``True`` if every pair of vectors is orthogonal.
    """
    for i, a in enumerate(args):
        for b in args[i + 1 :]:
            if not is_orthogonal(a, b):
                return False
    return True


def are_in_plane(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> bool:
    """Check whether three vectors are coplanar (scalar triple product close to
    zero)."""
    return jnp.isclose(jnp.dot(jnp.cross(a, b), c), 0.0)


def is_parallel(a: ArrayLike, b: ArrayLike) -> bool:
    """Check whether two vectors are parallel."""
    return jnp.isclose(a @ b, jnp.linalg.norm(a) * jnp.linalg.norm(b))


def is_normalized(a: ArrayLike) -> bool:
    """Check whether a vector has unit norm."""
    return jnp.isclose(jnp.linalg.norm(a), 1.0)


def rotate_about_point(
    input: Float[Array, "3"],
    rotation: Float[Array, "3 3"],
    center_of_rotation: Float[Array, "3"],
) -> Float[Array, "3"]:
    """Rotate a 3D point about a given center of rotation.

    Translates the point so the center of rotation is at the origin, applies
    the rotation, then translates back.

    Args:
        input: The 3D point to rotate.
        rotation: A ``(3, 3)`` rotation matrix.
        center_of_rotation: The 3D point to rotate around.

    Returns:
        The rotated 3D point.
    """
    return rotation @ (input - center_of_rotation) + center_of_rotation


def three_dimensional_representation_to_matrix(representation: Float[Array, "... 3"]) -> Float[Array, "... 3 3"]:
    """Converts a 3D rotation representation (with fixed z-axis) to a 3x3
    rotation matrix.

    Based on:
    "On the continuity of rotation representations in neural networks"
    Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li.
    Conference on Neural Information Processing Systems (NeurIPS) 2019.
    Args:
        representation: A shape (..., 6) array representing the rotation as a 6D vector.
    Returns:
        A shape (..., 3, 3) array representing the rotation as a 3x3 matrix.
    """
    representation = jnp.atleast_1d(representation)
    # original_shape = representation.shape

    x = representation

    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    z = jnp.array([0, 0, 1.0])
    y = jnp.cross(z, x)

    rotation_matrix = jnp.stack((x, y, z), axis=-2)
    return rotation_matrix
