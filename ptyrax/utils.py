import logging
import operator
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import equinox as eqx
import gin
import h5py
import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.ndimage import map_coordinates
from jax.tree_util import DictKey, GetAttrKey, SequenceKey
from jaxtyping import Array, ArrayLike, Complex, Float, Inexact, Integer, PyTree, Shaped
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ptyrax.parametrizations import ArrayParametrization

# region Plotting utilities


@jax.jit
def hsv_to_rgb(h: Float[Array, "..."], s: Float[Array, "..."], v: Float[Array, "..."]) -> Float[Array, "... 3"]:
    """Convert HSV color values to RGB.

    Args:
        h: Hue channel in [0, 1].
        s: Saturation channel in [0, 1].
        v: Value channel in [0, 1].

    Returns:
        RGB array with last dimension of size 3.
    """
    h *= 6.0
    i = jnp.floor(h).astype(jnp.int32)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i = i % 6
    i = jnp.expand_dims(i, axis=-1)
    conditions = [
        (i == 0, jnp.stack([v, t, p], axis=-1)),
        (i == 1, jnp.stack([q, v, p], axis=-1)),
        (i == 2, jnp.stack([p, v, t], axis=-1)),
        (i == 3, jnp.stack([p, q, v], axis=-1)),
        (i == 4, jnp.stack([t, p, v], axis=-1)),
        (i == 5, jnp.stack([v, p, q], axis=-1)),
    ]

    rgb = jnp.select([cond[0] for cond in conditions], [cond[1] for cond in conditions])
    return rgb


def complex_to_rgb(
    im: Complex[Array, "... w h"],
    log10: bool = False,
    cmap: Literal["hsv", "lab"] = "hsv",
    gamma: float = 1.0,
    clim: tuple[float, float] = None,
    scale_min: bool = False,
    max: float = 1.0,
) -> Float[Array, "... w h 3"]:
    """Convert a complex-valued image to an RGB representation.

    Encodes magnitude as brightness and phase as hue using an HSV colormap.

    Args:
        im: Complex input image.
        log10: Apply log10 to magnitude before mapping.
        cmap: Colormap style (currently only ``"hsv"`` supported).
        gamma: Gamma exponent applied to magnitude.
        clim: Optional (min, max) clipping range for magnitude.
        scale_min: Reserved (not used).
        max: Maximum RGB output value.

    Returns:
        RGB float array with values in [0, ``max``].
    """
    amplitude = jnp.abs(im)
    max_amplitude = jnp.amax(amplitude)
    amplitude /= max_amplitude
    if log10:
        amplitude = jnp.log10(amplitude)
    else:
        amplitude **= gamma

    if clim is not None:
        amplitude = jnp.maximum(clim[0], amplitude)
        amplitude = jnp.minimum(clim[1], amplitude)

    phase = jnp.angle(im)

    h = (phase + jnp.pi) / (2 * jnp.pi)  # Normalize to [0, 1]
    s = jnp.ones_like(h)
    v = amplitude

    img_rgb = hsv_to_rgb(h, s, v)
    return img_rgb * max


def plot(
    im: ArrayLike,
    show: bool = False,
    gs: Optional[GridSpec] = None,
    fig: Optional[Figure] = None,
    dpi: Optional[int] = 150,
    plot_text: bool = False,
    **kwargs,
) -> tuple[Figure, SubplotSpec, list[AxesImage]]:
    """Plot an image array, dispatching to complex or real plotting.

    Supports complex-valued images (phase-magnitude colormap), real-valued
    images, and objects with a ``__plot__`` method.

    Args:
        im: Image array to plot.
        show: If True, call ``plt.show()``.
        gs: Optional GridSpec for subplot placement.
        fig: Optional existing Figure.
        dpi: Figure DPI.
        plot_text: If True, annotate the figure with parameter info.
        **kwargs: Extra arguments forwarded to the plotting backend.

    Returns:
        Tuple of ``(fig, gs, image_artists)``.
    """
    if hasattr(im, "__plot__"):
        return im.__plot__(show=show, gs=gs, fig=fig, **kwargs)
    if gs is None:
        aspect = im.shape[0] if len(im.shape) > 2 else 1
        fig = plt.figure(dpi=dpi, figsize=(6, aspect * 4))
        gs = fig.add_gridspec(1, 1)[0]
    if isinstance(im, jax.Array) and im.shape[-1] > 512 and im.shape[-2] > 512:
        leading_dims = im.shape[:-2]
        im = jax.image.resize(im, (*leading_dims, 512, 512), method="nearest")
        # im = np.array(im)

    if im.dtype in ("complex64", "complex128"):
        fig, gs, image = plot_complex(im, fig, gs, **kwargs)
    else:
        fig, gs, image = plot_real(im, fig, gs, **kwargs)
    if show:
        plt.show()
    if plot_text:
        text = f"gamma: {kwargs.pop('gamma', 1.0)}"
        fig.text(0.8, 0.8, text, fontsize=10, va="center", ha="left", bbox=dict(facecolor="white", alpha=0.6))
    return fig, gs, image


@gin.configurable
def plot_complex(
    im: Complex[Array, "..."],
    fig: Figure,
    gs: SubplotSpec,
    log10: bool = False,
    cmap: str = "hsv",
    gamma: float = 1,
    title: Optional[str] = None,
    clim: Optional[tuple[float, float]] = None,
    cbar: Optional[bool] = None,
    **kwargs,
) -> tuple[Figure, SubplotSpec, list[AxesImage]]:
    """Plot a complex-valued image by converting it to RGB using a phase-
    magnitude colormap.

    Args:
        im: Complex image.
        fig: Matplotlib Figure to draw on.
        gs: GridSpec/SubplotSpec to place the image.
        log10: Whether to take log10 of magnitude.

    Returns:
        Tuple of (fig, gs, image artists).
    """
    converted_im = complex_to_rgb(im, log10, cmap, gamma, clim)
    return _plot_im(converted_im, fig, gs, title, clim, cbar=cbar, **kwargs)


@gin.configurable
def plot_real(
    im: Float[Array, "..."],
    fig: Figure,
    gs: SubplotSpec,
    title: str = None,
    cmap: str | None = None,
    gamma: float = 1.0,
    cbar: bool = False,
    log10: bool = False,
    epsilon: float = 1e-10,
    **kwargs,
) -> tuple[Figure, SubplotSpec, list[AxesImage]]:
    """Plot a real-valued image with optional gamma correction and log scaling.

    Args:
        im: Real-valued image.
        fig: Matplotlib Figure to draw on.
        gs: GridSpec/SubplotSpec to place the image.
        gamma: Gamma exponent to apply to the image.
        log10: Whether to apply log10 scaling (uses `epsilon` to avoid log(0)).

    Returns:
        Tuple of (fig, gs, image artists).
    """
    converted_im = im**gamma
    if log10:
        converted_im = np.log10(converted_im + epsilon)
    return _plot_im(converted_im, fig, gs, cmap=cmap, cbar=cbar, **kwargs)


def _plot_im(
    im: Inexact[Array, "..."],
    fig: Figure,
    gs: SubplotSpec,
    title: str = None,
    mode: str = "cols",
    cbar: bool = False,
    extent: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> tuple[Figure, SubplotSpec, list[AxesImage]]:
    def is_plottable_image(im: ArrayLike) -> bool:
        return len(im.shape) == 2 or (len(im.shape) == 3 and im.shape[-1] == 3)

    if not is_plottable_image(im):
        inner_gs = gs.subgridspec(1, im.shape[0]) if mode == "cols" else gs.subgridspec(im.shape[0], 1)
        images = []
        for i, sub_im in enumerate(im):
            sub_extent = extent[i] if extent is not None else None
            _, _, im = _plot_im(sub_im, fig, inner_gs[i], title=title, cbar=cbar, extent=sub_extent, **kwargs)
            images.append(im)
        return fig, gs, images
    ax = fig.add_subplot(gs)
    im = im.swapaxes(0, 1)  # First index: x = horizontal. Second index: y = vertical.
    image = ax.imshow(im, extent=extent, **kwargs)
    if extent is None:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        ax.title.set_text(title)
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=cax)
    return fig, gs, [image]


def real_to_rgb(
    tensor: Float[Array, "..."], log10: bool = False, cmap: str = "magma", gamma: float = 1.0, **kwargs
) -> Float[Array, "... 3"]:
    """Map a real-valued array to an RGB image via a matplotlib colormap.

    Args:
        tensor: Real input array.
        log10: Apply log10 scaling.
        cmap: Matplotlib colormap name.
        gamma: Gamma exponent applied to normalized amplitude.

    Returns:
        RGB float array with shape ``(*tensor.shape, 3)``.
    """
    amplitude = np.abs(tensor)
    if log10:
        amplitude[amplitude == 0] = np.amin(amplitude[amplitude > 0])
        amplitude = np.log10(amplitude)
    if np.amax(amplitude) != np.amin(amplitude):
        amplitude -= np.amin(amplitude)
        amplitude /= np.amax(amplitude)
        amplitude = amplitude**gamma
    else:
        amplitude = np.abs(np.ones_like(tensor))

    cmap = colormaps[cmap]
    img_rgb = cmap(amplitude)
    img_rgb = img_rgb[:, :, [0, 1, 2]]

    return img_rgb


# endregion
# region Array manipulation utilities
@gin.configurable
def compute_center_of_mass_shift(
    img: Float[Array, "... n m"],
    order: int = 1,
) -> Float[Array, "2"]:
    """Compute the center-of-mass shift required to center an image.

    Args:
        img: Input image (can be batched); non-negative values are assumed.
        order: Power to raise pixel values when computing the weighted center.

    Returns:
        Shift vector (dy, dx) to translate the image to the center.
    """
    # img ≥ 0 assumed.  Compute current COM:
    img = jnp.abs(img)
    coords = jnp.indices(img.shape)
    mass = (img**order).sum()
    com = (coords * img**order).sum(axis=(-2, -1)) / mass  # shape (2,)
    center = (jnp.array(img.shape) - 1) / 2  # zero‐based pixel center
    shift = center - com  # how much to translate
    return shift


@gin.configurable("sliced_shift")
def slice_at_center_to_shape(
    x: Float[Array, "... n m"],
    center: Float[Array, "2"],
    target_shape: tuple[int, int],
) -> Float[Array, "... n m"]:
    """Extract a centered slice of `x` with the given `target_shape` around
    `center`.

    Args:
        x: Array from which to slice. Last two dimensions are spatial.
        center: Center coordinate (y, x) in the same coordinate system as `x`.
        target_shape: Desired output spatial shape (height, width).

    Returns:
        Dynamically sliced view of `x` with spatial dims equal to `target_shape`.
    """
    (w, h) = (x.shape[-2], x.shape[-1])
    corner = center - jnp.array(target_shape) // 2
    coords = jnp.array((w / 2 + corner[0], h / 2 + corner[1]), dtype=int)
    prefix_axes_coords = (0,) * (len(x.shape) - 2)
    prefix_axes_shapes = x.shape[:-2]
    return lax.dynamic_slice(x, (*prefix_axes_coords, *coords), (*prefix_axes_shapes, *target_shape))


def zero_pad_to_shape(
    x: Float[Array, "... n m"],
    target_shape: tuple[int, int],
) -> Float[Array, "... n m"]:
    """Symmetrically zero-pad an array to ``target_shape``.

    Args:
        x: Input array (last two dims are spatial).
        target_shape: Desired spatial dimensions.

    Returns:
        Zero-padded array with spatial dimensions equal to ``target_shape``.
    """
    new_array = np.zeros(target_shape, dtype=x.dtype)
    (w, h) = (x.shape[-2], x.shape[-1])
    dw, dh = ((target_shape[-2] - w) // 2, (target_shape[-1] - h) // 2)
    new_array[..., dw : dw + w, dh : dh + h] = x
    return new_array


def shift_image(img: jnp.ndarray, shift: jnp.ndarray, order: int = 1) -> jnp.ndarray:
    """Shift a 2D image using scipy.ndimage.map_coordinates with zero padding.

    Parameters:
        img   : 2D NumPy array
        shift : Tuple or array (dy, dx), the shift to apply
        order : Interpolation order (1 = bilinear, 3 = cubic, etc.)

    Returns:
        Shifted image with same shape as input, zero-padded at edges.
    """
    if img.ndim != 2:
        raise ValueError(f"Input image must be 2D. Got {img.ndim}D instead.")
    # A hack which allows the shift to be higher dimensional
    # (used when the amount of probe modes does not match the sample modes)
    dy, dx = shift[0, :, 0] if len(shift.shape) == 3 else shift[:, 0] if len(shift.shape) == 2 else shift

    ny, nx = img.shape

    # Coordinates of the output image grid
    y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
    coords = (y - dy, x - dx)

    # Interpolate from shifted coordinates
    shifted = map_coordinates(
        img,
        coords,
        order=order,
        mode="constant",
        cval=0.0,  # zero padding
    )
    return shifted


def tile_axes(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Tile an array to at least ``target_shape`` and center-crop to exact
    size.

    Args:
        arr: Input array to tile.
        target_shape: Desired output shape.

    Returns:
        Array tiled and cropped to ``target_shape``.
    """
    reps = [max(1, -(-t // s)) for s, t in zip(arr.shape, target_shape)]
    arr = np.tile(arr, reps)
    slices = tuple(
        slice((arr.shape[i] - target_shape[i]) // 2, (arr.shape[i] - target_shape[i]) // 2 + target_shape[i])
        for i in range(arr.ndim)
    )
    return arr[slices]


def center_crop(arr: np.ndarray, target_len: int, axis: int) -> np.ndarray:
    """Crop an array symmetrically along a single axis.

    Args:
        arr: Input array.
        target_len: Desired length along ``axis``.
        axis: Axis to crop.

    Returns:
        Center-cropped array.
    """
    src_len = arr.shape[axis]
    start = (src_len - target_len) // 2
    end = start + target_len
    return np.take(arr, indices=range(start, end), axis=axis)


def repeat_axis(arr: np.ndarray, target_len: int, axis: int) -> np.ndarray:
    """Repeat elements along an axis and center-crop to ``target_len``.

    Args:
        arr: Input array.
        target_len: Desired length along ``axis``.
        axis: Axis to expand.

    Returns:
        Repeated and center-cropped array.
    """
    src_len = arr.shape[axis]
    reps = -(-target_len // src_len)  # ceil division
    arr = np.repeat(arr, reps, axis=axis)
    return center_crop(arr, target_len, axis)


def center_pad(arr: np.ndarray, target_len: int, axis: int) -> np.ndarray:
    """Symmetrically zero-pad an array along a single axis.

    Args:
        arr: Input array.
        target_len: Desired length along ``axis``.
        axis: Axis to pad.

    Returns:
        Zero-padded array.
    """
    src_len = arr.shape[axis]
    pad_before = (target_len - src_len) // 2
    pad_after = target_len - src_len - pad_before
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad_before, pad_after)
    return np.pad(arr, pad_width, mode="constant")


def resize_to_match(
    arr: np.ndarray, target_shape: Tuple[int, ...], axis_policies: Dict[int, str] = None, default_policy: str = "pad"
) -> np.ndarray:
    """
    Resize arr to target_shape:
    - Contracting axes: crop symmetrically
    - Expanding axes: per-axis policy or default_policy
    """
    if axis_policies is None:
        axis_policies = {}
    result = arr
    for axis, (src_len, dst_len) in enumerate(zip(result.shape, target_shape)):
        if src_len == dst_len:
            continue
        if src_len > dst_len:
            result = center_crop(result, dst_len, axis)
        else:  # src_len < dst_len
            policy = axis_policies.get(axis, default_policy)
            if policy == "pad":
                result = center_pad(result, dst_len, axis)
            elif policy == "repeat":
                result = repeat_axis(result, dst_len, axis)
            elif policy != "tile":
                raise ValueError(f"Unknown expand policy '{policy}' for axis {axis}")
    # If any axis requested tile, apply global tiling at the end
    if "tile" in axis_policies.values() or default_policy == "tile":
        result = tile_axes(result, target_shape)
    return result


def convert_to_ft_sampling(
    pixel_number: Integer[Array, "... d"],
    pixel_size: Integer[Array, "... d"],
    scaling_factor: float = 1.0,
    prop_dist: float = None,
    wavelength: float = None,
) -> tuple[Integer[Array, "... d"], Float[Array, "... d"]]:
    """Compute the reciprocal-space (Fourier) sampling parameters.

    Given real-space pixel count and size, returns the corresponding Fourier
    pixel count and size. Optionally computes the scaling from wavelength and
    propagation distance.

    Args:
        pixel_number: Number of pixels in each dimension.
        pixel_size: Pixel size in each dimension.
        scaling_factor: Manual scaling factor (mutually exclusive with
            ``wavelength``/``prop_dist``).
        prop_dist: Propagation distance (requires ``wavelength``).
        wavelength: Photon wavelength (requires ``prop_dist``).

    Returns:
        Tuple of ``(ft_pixel_number, ft_pixel_size)``.

    Raises:
        ValueError: If arguments are inconsistent.
    """
    if wavelength is not None and prop_dist is not None:
        if scaling_factor != 1.0:
            raise ValueError(
                "Error computing the Fourier sampling. If wavelength and prop_dist are set, then scaling "
                "factor cannot be set"
            )
        else:
            scaling_factor = prop_dist * wavelength
    elif wavelength is not None or prop_dist is not None:
        raise ValueError(
            "Error computing the Fourier sampling. Only one of wavelength and prop_dist were set, "
            "while both are required."
        )

    ft_pixel_size = scaling_factor / (pixel_number * pixel_size)

    return np.array(pixel_number), ft_pixel_size


def adjoint(f: Shaped[Array, "... m n"]) -> Shaped[Array, "... m n"]:
    """Compute the conjugate transpose of an array.

    Args:
        f: Input array.

    Returns:
        Conjugate transpose (swapping last two axes).
    """
    return jnp.conjugate(jnp.swapaxes(f, -2, -1))


def orthogonal(f: Shaped[Array, "... d mn"]) -> Shaped[Array, "... d mn"]:
    """Orthogonalize rows of a matrix using SVD.

    Args:
        f: Input matrix or batch of matrices.

    Returns:
        Matrix with orthogonalized rows.
    """
    s, u, v = jnp.linalg.svd(jnp.matmul(f, adjoint(f)))
    f_orthogonal = jnp.matmul(adjoint(v), f)
    return f_orthogonal


parallel_orthogonal = jax.vmap(orthogonal, in_axes=(0,))


def orthogonalize(f: Shaped[Array, "... m n"]) -> Shaped[Array, "... m n"]:
    """Orthogonalize probe modes in a multi-mode array.

    Args:
        f: Array of shape ``(modes, m, n)`` representing probe modes.

    Returns:
        Orthogonalized probe modes with same shape.

    Raises:
        NotImplementedError: If input has additional wavelength dimensions.
        ValueError: If no mode dimension is present.
    """
    if len(f.shape) == 3:  # probe mode dim only
        f_flat = jnp.reshape(f, [f.shape[0], f.shape[1] * f.shape[2]])
        f_flat_orthogonal = orthogonal(f_flat)
    elif len(f.shape) == 4:  # also wavelength dim
        raise NotImplementedError("Additional probe dimensions not yet implemented")
    else:
        raise ValueError("No dimension to orthogonalize over!")

    f_orthogonal = np.reshape(f_flat_orthogonal, f.shape)
    return f_orthogonal


@gin.configurable
def load_hdf5(file_path: str, key_translation: dict = None) -> dict:
    """This function loads the data in a hdf5 file from file_path and returns a
    dictionary.

    The data_type =  all, params, ptychogram
    """
    with h5py.File(file_path, "r") as file:
        keys = list(file.keys())
        data = {}
        for key in keys:
            if key_translation:
                try:
                    data[key] = file[key_translation[key]][()]
                except KeyError:
                    logging.warning(
                        f"Could not find hfd5 key {key} in the key_translation dictionary. Returning the key as is."
                    )
                    data[key] = file[key][()]
            else:
                data[key] = file[key][()]
    return data


def sort_images_by_time(image_paths: list[str]) -> list[str]:
    """Sort image file paths by their filesystem modification time.

    Args:
        image_paths: List of file paths to sort.

    Returns:
        Paths sorted by ascending modification time.
    """

    def extract_timestamp(image_path: str) -> datetime:
        # Load the timestamp from the OS file metadata
        timestamp = os.path.getmtime(image_path)
        # Convert the timestamp to a datetime object
        return datetime.fromtimestamp(timestamp)

    sorted_image_paths = sorted(image_paths, key=extract_timestamp)
    return sorted_image_paths


def save_hdf5(file_path: str, data: dict) -> None:
    """This function saves the data (a dictionary) to a hdf5 file at
    file_path."""
    with h5py.File(file_path, "w") as file:
        for key in list(data.keys()):
            if data[key] is None:
                continue
            file.create_dataset(key, data=data[key])


def center_scan_pos(scan_pos: Float[Array, "N d"]) -> Float[Array, "N d"]:
    """Center scanning positions by subtracting their mean.

    Args:
        scan_pos: Array of scanning positions.

    Returns:
        Mean-subtracted positions.
    """
    return scan_pos - np.mean(scan_pos, axis=0)


def normalize(a: Shaped[Array, "..."], axes: tuple[int, "..."] = None) -> Shaped[Array, "..."]:
    """Normalize an array by its maximum value along specified axes.

    Args:
        a: Input array.
        axes: Axes over which to take the maximum.

    Returns:
        Array divided by its maximum.
    """
    return a / jnp.max(a, axis=axes)


def normalize_power(a: Shaped[Array, "..."], axis: tuple[int, "..."] = (0, 1)) -> Shaped[Array, "..."]:
    r"""Normalize an array so its total power equals 1.

    The normalization enforces :math:`\sum |a|^2 = 1` over ``axis``.

    Args:
        a: Input array.
        axis: Axes over which to compute the power.

    Returns:
        Power-normalized array.
    """
    return a / jnp.sqrt(jnp.sum(jnp.abs(a) ** 2, axis=axis))


# region Jax utilities
def vmap_nested(fn: Callable, in_axes: tuple[int, "..."], *args, **kwargs) -> Callable:
    """Apply nested :func:`jax.vmap` for multiple batch dimensions.

    Args:
        fn: Function to vectorize.
        in_axes: Tuple of axes—one per nesting level.

    Returns:
        Multi-level vmapped function.
    """
    if len(in_axes) == 1:
        return jax.vmap(fn, in_axes=in_axes[0], *args, **kwargs)
    return jax.vmap(vmap_nested(fn, in_axes=in_axes[1:], *args, **kwargs), in_axes=in_axes[0], *args, **kwargs)


def reduced_grad(fn: Callable, reduction_fn: Callable = lambda *args: jnp.sum(jnp.abs(*args))) -> Callable:
    """Create a gradient function with a scalar reduction applied first.

    Args:
        fn: Function whose output will be reduced then differentiated.
        reduction_fn: Scalar reduction (default: sum of absolute values).

    Returns:
        Gradient function of the reduced output.
    """

    def reduced(*args, **kwargs) -> Shaped[Array, "..."]:
        a = fn(*args, **kwargs)
        return reduction_fn(a)

    return jax.grad(reduced)


def count_parameters(pytree: PyTree) -> int:
    """Counts the total number of parameters in a PyTree.

    Args:
        pytree: A PyTree containing parameters.

    Returns:
        int: Total number of parameters.
    """

    # Function to return the size of each leaf node
    def leaf_size(leaf: Shaped[Array, "..."]) -> int:
        return leaf.size

    total_parameters = jax.tree.reduce(operator.add, jax.tree.map(leaf_size, pytree))
    return total_parameters


def tree_slice_first(tree: PyTree) -> PyTree:
    """Slices the first element from each leaf in a PyTree.

    Args:
        tree: A PyTree containing arrays.
    Returns:
        PyTree: A new PyTree with the first element sliced from each leaf.
    """

    def is_leaf(x: Any):  # noqa: ANN401
        return isinstance(x, jnp.ndarray) and x.ndim > 0 or x is None

    def slice_first_or_none(x: Any):  # noqa: ANN401
        return x[0] if x is not None else x

    return jax.tree.map(slice_first_or_none, tree, is_leaf=is_leaf)


def unstack_tree(stacked: PyTree) -> list[PyTree]:
    """Split a stacked PyTree (with leading batch axis) into a list of PyTrees.

    Args:
        stacked: PyTree where every leaf has a leading batch dimension.

    Returns:
        List of PyTrees, one per element along axis 0.
    """
    # Infer N from the first leaf
    leaf0 = jax.tree.leaves(stacked)[0]
    N = leaf0.shape[0]

    # Build each tree by indexing axis 0
    return [jax.tree.map(lambda v: v[i], stacked) for i in range(N)]


def make_path_string(path: list[str]) -> str:
    """Convert a JAX key-path list to a dot-separated string.

    Args:
        path: List of JAX tree path elements.

    Returns:
        Dot-joined path string.
    """

    def _get_path_name(p: str) -> str:
        if isinstance(p, GetAttrKey):
            return p.name
        elif isinstance(p, DictKey):
            return p.key
        elif isinstance(p, SequenceKey):
            return f"{p.idx}"
        else:
            raise ValueError(f"Unknown path element type: {type(p)}")

    return ".".join([_get_path_name(p) for p in path])


# endregion
# region Functional utilities
@gin.configurable
def identity(*args: Any) -> Any:  # noqa: ANN401
    """Identity function that returns its arguments unchanged.

    Useful as a default no-op callback in gin-configurable pipelines.

    Args:
        *args: Any arguments.

    Returns:
        The input arguments as a tuple.
    """
    return args


def single_identity(args: tuple[Any, "..."]) -> Any:  # noqa: ANN401
    """Return a single-element tuple unchanged.

    Convenience identity function for callbacks that receive a tuple.

    Args:
        args: Input tuple.

    Returns:
        The same tuple.
    """
    return args


@gin.register()
def soft_clip(
    a: Shaped[Array, "..."],
    min_value: float | int,
    max_value: float | int,
    relu_like: Callable = jax.nn.identity,
    scale: float = 1.0,
) -> Shaped[Array, "..."]:
    """Differentiable soft-clipping function.

    Applies a smooth clamping operation using a ReLU-like activation
    to keep values within ``[min_value, max_value]``.

    Args:
        a: Input array.
        min_value: Lower clipping bound.
        max_value: Upper clipping bound.
        relu_like: Activation function for soft clamping.
        scale: Scaling factor applied before and after clipping.

    Returns:
        Clipped array.
    """
    a = a / scale
    # jax.debug.callback(lambda norm: print(f"a norm: {norm}"), jnp.linalg.norm(a))
    # jax.debug.callback(lambda norm: print(f"scale: {norm}"), scale)
    # jax.debug.callback(lambda norm: print(f"a min: {norm}"), jnp.min(a))
    # jax.debug.callback(lambda norm: print(f"a max: {norm}"), jnp.max(a))
    a = relu_like(a - min_value) + min_value
    a = -relu_like(-(a - max_value)) + max_value
    a = a * scale
    return a


def median_pixel_value(img: Float[Array, "..."]) -> float:
    """Compute the median pixel of an image."""
    return np.percentile(img, 50)


def fft(
    x: Inexact[Array, "..."],
    axes: tuple[int, int] = (-2, -1),
    norm: Literal["backward", "ortho", "forward"] = "ortho",
    fftshift: bool = True,
) -> Complex[Array, "..."]:
    """Compute centered 2D FFT.

    Applies ``ifftshift`` before and ``fftshift`` after the FFT so that
    the zero-frequency component is in the center.

    Args:
        x: Input array.
        axes: Axes over which to compute the FFT.
        norm: Normalization mode.
        fftshift: If True, center the transform.

    Returns:
        Complex FFT of the input.
    """
    if not fftshift:
        return jnp.fft.fft2(x, axes=axes, norm=norm)
    return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(x, axes=axes), axes=axes, norm=norm), axes=axes)


def ifft(
    x: Inexact[Array, "..."],
    axes: tuple = (-2, -1),
    norm: Literal["backward", "ortho", "forward"] = "ortho",
    fftshift: bool = True,
) -> Complex[Array, "..."]:
    """Compute centered 2D inverse FFT.

    Applies ``ifftshift`` before and ``fftshift`` after the IFFT so that
    the zero-frequency component is handled correctly.

    Args:
        x: Input array (frequency domain).
        axes: Axes over which to compute the IFFT.
        norm: Normalization mode.
        fftshift: If True, center the transform.

    Returns:
        Complex inverse FFT of the input.
    """
    if not fftshift:
        return jnp.fft.ifft2(x, axes=axes, norm=norm)
    return jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(x, axes=axes), axes=axes, norm=norm), axes=axes)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten nested dictionaries into a single-level dict.

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for keys at this level (used in recursion).
        sep: Separator joining nested key parts.

    Returns:
        Flat dictionary with concatenated keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


identity_transform = jnp.eye(4)


@gin.configurable
def scaled_mean(a: ArrayLike, scale: float = 1.0, **kwargs) -> Float[Array, ""]:
    """Compute the mean of an array scaled by a constant factor.

    Typically configured via gin as a loss reduction function.

    Args:
        a: Input array.
        scale: Multiplicative scaling applied after mean reduction.
        **kwargs: Extra arguments forwarded to :func:`jax.numpy.mean`.

    Returns:
        Scalar mean value times ``scale``.
    """
    return jnp.mean(a, **kwargs) * scale


def phase_only_exp(phase: Float[Array, "... m n"]) -> Complex[Array, "... m n"]:
    """Compute a unit-magnitude complex exponential from a real phase.

    Equivalent to ``exp(1j * phase)`` but avoids complex input.

    Args:
        phase: Real-valued phase array in radians.

    Returns:
        Complex array with unit magnitude and given phase.

    Raises:
        ValueError: If ``phase`` is already complex.
    """
    if jnp.iscomplexobj(phase):
        raise ValueError("Input to phase_only_exp should be real-valued phase array.")
    return jnp.cos(phase) + 1j * jnp.sin(phase)


def phase_only_exp_np(phase: Float[Array, "... m n"]) -> Complex[Array, "... m n"]:
    """NumPy version of :func:`phase_only_exp`.

    Args:
        phase: Real-valued phase array in radians.

    Returns:
        Complex NumPy array with unit magnitude.

    Raises:
        ValueError: If ``phase`` is already complex.
    """
    if np.iscomplexobj(phase):
        raise ValueError("Input to phase_only_exp should be real-valued phase array.")
    return np.cos(phase) + 1j * np.sin(phase)


def abs_sq(x: Shaped[Array, "..."]) -> Float[Array, "..."]:
    """Compute the squared absolute value without taking a square root.

    More efficient than ``jnp.abs(x)**2`` for complex inputs.

    Args:
        x: Input array (real or complex).

    Returns:
        Real-valued squared magnitude.
    """
    return (x * x.conj()).real


def make_length_n(parameter: float | int | np.ndarray | list | tuple, n: int = 2) -> np.array:
    """Utility to convert a single parameter value or a list of parameter
    values into an array of length n."""
    if isinstance(parameter, float | int):
        parameter = np.array([parameter] * n)
    if isinstance(parameter, tuple | list):
        parameter = np.array(parameter)
    if len(parameter) == 1:
        parameter = np.array([parameter[0]] * n)
    return np.array(parameter)


def normalize_hdf5_path(path: str | None) -> str:
    """Normalize an HDF5-like path by collapsing duplicate separators."""
    if not path:
        return ""
    cleaned = re.sub(r"/+", "/", path.strip())
    return cleaned.strip("/")


def join_hdf5_paths(prefix: str | None, suffix: str) -> str:
    """Join two HDF5-like paths while keeping a single separator."""
    normalized_prefix = normalize_hdf5_path(prefix)
    normalized_suffix = normalize_hdf5_path(suffix)
    if not normalized_prefix:
        return normalized_suffix
    if not normalized_suffix:
        return normalized_prefix
    return f"{normalized_prefix}/{normalized_suffix}"


def compile_policy_patterns(
    policy_map: dict[str, dict[str, Any]] | None,
) -> list[tuple[re.Pattern[str], dict[str, Any]]]:
    """Compile regex policies once and validate patterns up-front."""
    compiled: list[tuple[re.Pattern[str], dict[str, Any]]] = []
    if not policy_map:
        return compiled
    for pattern, policy in policy_map.items():
        try:
            compiled.append((re.compile(pattern), policy))
        except re.error as exc:
            raise ValueError(f"Invalid policy regex '{pattern}': {exc}") from exc
    return compiled


def wrap_like_parametrization(reference: Any, value: Any) -> Any:  # noqa: ANN401
    """Wrap a value in the same parametrization type as a reference.

    If ``reference`` is an ArrayParametrization, wraps ``value`` in the
    same subclass; otherwise returns ``value`` unchanged.

    Args:
        reference: Object whose type is inspected.
        value: Value to potentially wrap.

    Returns:
        Wrapped or unwrapped value.
    """
    if isinstance(reference, ArrayParametrization):
        return type(reference)(value)
    return value


def set_probe_data_preserve_parametrization(model: Any, new_probe_data: Any) -> Any:  # noqa: ANN401
    """Update model probe data while preserving the parametrization wrapper.

    Uses :func:`eqx.tree_at` to replace ``model.illumination.probe.data``
    with ``new_probe_data`` wrapped in the original parametrization type.

    Args:
        model: Model whose probe data should be updated.
        new_probe_data: New data to set.

    Returns:
        Updated model with new probe data.
    """
    wrapped_data = wrap_like_parametrization(model.illumination.probe.data, new_probe_data)
    return eqx.tree_at(lambda m: m.illumination.probe.data, model, wrapped_data)


def warn_if_duplicate_normalized_keys(state_keys: list[str], normalize_keys: bool) -> None:
    """Warn when key normalization collapses distinct checkpoint paths.

    If ``normalize_keys`` is enabled and stripping leading underscores from
    path segments causes two different keys to map to the same normalized
    key, a warning is emitted.

    Args:
        state_keys: List of HDF5 state keys.
        normalize_keys: Whether normalization is active.
    """
    if not normalize_keys:
        return

    seen: dict[str, str] = {}
    for key in state_keys:
        normalized = "/".join(part.lstrip("_") for part in key.split("/"))
        if normalized in seen and seen[normalized] != key:
            logging.warning(
                "HDF5 key normalization maps multiple paths to '%s': '%s' and '%s'.",
                normalized,
                seen[normalized],
                key,
            )
        else:
            seen[normalized] = key


def make_or_reuse_axes(fig: plt.Figure = None, gs: SubplotSpec = None) -> tuple[plt.Figure, plt.Axes]:
    """Create or reuse matplotlib Figure and Axes.

    Args:
        fig: Existing figure (required if ``gs`` is provided).
        gs: GridSpec subplot to add an axes to.

    Returns:
        Tuple of ``(fig, ax)``.

    Raises:
        ValueError: If ``gs`` is provided without ``fig``.
    """
    if gs is None:
        return plt.subplots(1, 1)
    else:
        if fig is None:
            raise ValueError("If gs is provided, fig must also be provided.")
        ax = fig.add_subplot(gs)
        return fig, ax
