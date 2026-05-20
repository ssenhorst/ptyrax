import functools
import logging
import pathlib
from typing import Callable, Literal

# import chromatix
import gin
import jax
import jax.numpy as jnp
import numpy as np
from imageio.v3 import imread
from jaxtyping import Array, ArrayLike, Complex, Float, Key
from skimage.transform import rescale, resize

from ptyrax.optics import absorption_coefficient, reflection_coefficient, transmission_coefficient
from ptyrax.spatial import CoordinateSystem, SamplingGrid, interpolate_grid_to_grid, shift_with_interpolation
from ptyrax.utils import (
    fft,
    ifft,
    load_hdf5,
    make_length_n,
    phase_only_exp,
    phase_only_exp_np,
    resize_to_match,
    slice_at_center_to_shape,
    zero_pad_to_shape,
)


@gin.configurable
def binary_complex_image(
    sampling: SamplingGrid,
    image_path: pathlib.Path | str,
    image_pixel_size: float = 1.0,
    scale: float = 1.0,
    image_shift: tuple[float, float] = (0.0, 0.0),
    threshold: float = 0.5,
    high_amplitude: float = 1.0,
    high_phase: float = 0.0,
    low_amplitude: complex = 0.0,
    low_phase: complex = 0.0,
    normalize: bool = False,
) -> Complex[Array, " m n"]:
    """Load an image and produce a binary complex-valued initializer.

    Args:
        sampling: Target sampling grid for the initializer.
        image_path: Path to the input image.
        image_pixel_size: Pixel size of the input image.
        image_shift: (x,y) shift to apply when mapping to the target grid.
        threshold: Threshold to binarize the image.

    Returns:
        Complex array shaped to `sampling` representing the binary initializer.
    """

    image = imread(image_path)
    if image.ndim == 3:
        image = jnp.mean(image, axis=-1)
    image = image.T  # Transpose to match coordinate system conventions (x is first axis)
    image = image / jnp.max(image)
    image_sampling = SamplingGrid.from_tuples(image.shape, (image_pixel_size / scale, image_pixel_size / scale))
    if sampling is None:
        sampling = image_sampling

    image_data = interpolate_grid_to_grid(
        image,
        image_sampling,
        CoordinateSystem(translation=jnp.array((image_shift[0] * scale, image_shift[1] * scale, 0.0))),
        sampling,
        CoordinateSystem(),
        interpolation_mode="amplitude_phase",
    )
    if image_data.shape != sampling.shape:
        raise ValueError(f"Input image shape {image_data.shape} does not match target shape {sampling.shape}")
    high_value = high_amplitude * phase_only_exp(high_phase)
    low_value = low_amplitude * phase_only_exp(low_phase)
    high_value = jnp.ones(sampling.shape, dtype=jnp.complex64) * high_value
    low_value = jnp.ones(sampling.shape, dtype=jnp.complex64) * low_value

    binary_image_data = jnp.where(image_data >= threshold, high_value, low_value)
    if normalize:
        binary_image_data = binary_image_data / jnp.linalg.norm(binary_image_data)
    return binary_image_data


@gin.configurable
def binary_reflection_image(
    sampling: SamplingGrid,
    image_path: pathlib.Path | str,
    image_pixel_size: float = 1.0,
    image_shift: tuple[float, float] = (0.0, 0.0),
    threshold: float = 0.5,
    high_refl_material: str = "Al",
    high_trans_material: str = None,
    low_refl_material: str = "Si",
    low_trans_material: str = None,
    wavelength: float = 1.0,  # meters
    angle_of_incidence: float = 0.0,
    thickness: float = 0.0,
    thickness_wavelength_units: bool = False,
    polarization: Literal["s", "p"] = "p",
) -> Complex[Array, " m n"]:
    """Build a binary complex reflection-image initializer from an input image.

    Args:
        sampling: Target sampling grid.
        image_path: Path to the input image.
        image_pixel_size: Pixel size of the input image.
        threshold: Threshold to binarize.
        high_refl_material/low_refl_material: Material identifiers for reflection coefficients.

    Returns:
        Complex array shaped to `sampling` representing reflection coefficients.
    """

    if thickness_wavelength_units:
        thickness *= wavelength

    def get_reflection_coefficient(t_mat: str, r_mat: str) -> complex:
        if t_mat is None and r_mat is None or r_mat in {"vacuum", "air"}:
            return 0.0
        r = reflection_coefficient(
            from_material=t_mat if t_mat is not None else "vacuum",
            to_material=r_mat,
            wavelength=wavelength,
            angle_of_incidence=angle_of_incidence,
            polarization=polarization,
        )
        t = (
            transmission_coefficient(
                from_material="vacuum",
                to_material=t_mat,
                wavelength=wavelength,
                angle_of_incidence=angle_of_incidence,
                polarization=polarization,
            )
            if t_mat is not None
            else 1.0
        )
        a = (
            absorption_coefficient(
                material=t_mat,
                wavelength=wavelength,
                angle_of_incidence=angle_of_incidence,
                z=thickness,
            )
            if t_mat is not None
            else 1
        )
        return t * a * r * a * t

    high_coefficient = get_reflection_coefficient(high_trans_material, high_refl_material)
    low_coefficient = get_reflection_coefficient(low_trans_material, low_refl_material)

    logging.info(
        f"Binary reflection image initialized with {high_refl_material=}: ({high_coefficient:.2e}) "
        f"abs: {jnp.abs(high_coefficient):.2e}, phase: {jnp.angle(high_coefficient, deg=True):3.0f},"
        f" {low_refl_material=}: ({low_coefficient:.2e}) "
        f"abs: {jnp.abs(low_coefficient):.2e}, phase: {jnp.angle(low_coefficient, deg=True):3.0f},"
        f"relative abs: {jnp.abs(high_coefficient / low_coefficient):.2e}: "
        f"phase difference: {jnp.angle(high_coefficient / low_coefficient, deg=True):3.0f}, "
    )

    high_coefficient = jnp.ones(sampling.shape, dtype=jnp.complex64) * high_coefficient
    low_coefficient = jnp.ones(sampling.shape, dtype=jnp.complex64) * low_coefficient

    return binary_complex_image(
        sampling=sampling,
        image_path=image_path,
        image_pixel_size=image_pixel_size,
        image_shift=image_shift,
        threshold=threshold,
        high_amplitude=np.abs(high_coefficient),
        high_phase=np.angle(high_coefficient),
        low_amplitude=np.abs(low_coefficient),
        low_phase=np.angle(low_coefficient),
    )


@gin.configurable
def from_test_images(
    sampling: SamplingGrid,
    amplitude_image: str = "camera",
    phase_image: str = "astronaut",
) -> Complex[Array, " m n"]:
    """Create a complex initializer from two test images (amplitude and phase).

    Args:
        sampling: Target sampling grid.
        amplitude_image: Name of skimage test image to use for amplitude.
        phase_image: Name of skimage test image to use for phase.

    Returns:
        Complex array shaped to `sampling`.
    """

    import skimage.data

    amplitude_image = getattr(skimage.data, amplitude_image)()
    phase_image = getattr(skimage.data, phase_image)()
    if amplitude_image.ndim == 3:
        amplitude_image = jnp.mean(amplitude_image, axis=-1)
    amplitude_image = resize_to_match(amplitude_image, sampling.shape)
    if phase_image.ndim == 3:
        phase_image = jnp.mean(phase_image, axis=-1)
    phase_image = resize_to_match(phase_image, sampling.shape)
    amplitude_image = amplitude_image / jnp.max(amplitude_image)
    phase_image = phase_image / jnp.max(phase_image) * 2 * jnp.pi - np.pi
    return amplitude_image * phase_only_exp(phase_image)


@gin.register()
def probe_initializer_from_gt(
    sampling: SamplingGrid,
    pixel_size: ArrayLike,
    dtype: jnp.dtype = jnp.complex64,
) -> np.ndarray:
    """Initialize a probe field from a ground-truth HDF5 file.

    Loads probe data from ``data/SI0007_ground_truth.hdf5``, rescales to match
    the target pixel size, and crops or pads to fit the sampling grid.

    Args:
        sampling: Target sampling grid.
        pixel_size: Desired pixel size for the output probe.
        dtype: Data type for the returned array.

    Returns:
        Complex probe array shaped to ``sampling``.
    """
    shape = sampling.shape
    data = load_hdf5("data/SI0007_ground_truth.hdf5")
    roi_unit = 1e-6
    sim_unit = 1

    probe = data["probe"][0, 0]
    probe_pixel_size = data["probe_pixel_size"]
    relative_pixel_size = (probe_pixel_size * roi_unit) / (np.array(pixel_size) * sim_unit)
    logging.info(f"{relative_pixel_size=}")

    probe = rescale(probe.real, relative_pixel_size) + 1j * rescale(probe.imag, relative_pixel_size)
    probe = probe.T
    probe = probe.astype(dtype)
    dw, dh = (shape[0] - probe.shape[0], shape[1] - probe.shape[1])
    if dw > 0:
        probe = zero_pad_to_shape(probe, (shape[0], probe.shape[1]))
    if dh > 0:
        probe = zero_pad_to_shape(probe, (probe.shape[0], shape[1]))
    try:
        probe = slice_at_center_to_shape(probe, np.array([0, 0]), shape)
    except TypeError:
        logging.info(f"zero padding from shape {probe.shape} to shape {shape}")
        probe = zero_pad_to_shape(probe, shape)
    return probe


@gin.configurable
def aperture(
    sampling: SamplingGrid,
    radius: float = None,
    dtype: jnp.dtype = jnp.complex64,
    scale: complex = 1.0 + 0j,
    normalize: bool = True,
    defocus: float = 0.0,
    **kwargs,
) -> Array:
    """Generates a  corresponding to an aperture.

    Args:
        shape (tuple[int, ...]): The shape of the field.
        radius (Union[None, float, jnp.ndarray]): The radius of the aperture.
        dtype (jnp.dtype): The data type of the field.
        scale (Union[None, complex, float]): scales the field by a constant multiplier.
        **kwargs: Additional keyword arguments.

    Returns:
        The generated aperture field as an (M, N, 1) jax array.
    """
    shape = sampling.shape
    base = np.ones(shape, dtype=dtype)
    if radius is None:
        radius = np.array(shape) / 8
    radius = make_length_n(radius)
    x, y = np.meshgrid(
        np.arange(shape[0]) - shape[0] / 2,
        np.arange(shape[1]) - shape[1] / 2,
        indexing="ij",
    )
    coords = np.stack((x, y), axis=0)
    radius = radius[:, np.newaxis, np.newaxis]
    stretched_coords = coords / radius
    stretched_r2 = jnp.sum(stretched_coords**2, axis=0)
    output = base * (stretched_r2 < 1)
    if defocus not in [0.0, (0.0, 0.0)]:
        defocus = np.array(defocus)[:, np.newaxis, np.newaxis]
        shape = np.array(shape)[:, np.newaxis, np.newaxis]
        stretched_r2 = np.sum((coords / radius * defocus / shape) * coords, axis=0)
        exponential = phase_only_exp_np(np.pi * stretched_r2)
        output = output * exponential
    output = output * scale
    return output / jnp.linalg.norm(output) if normalize else output


@gin.configurable
def gaussian(
    sampling: SamplingGrid,
    std: float = None,
    radius: float = None,
    noise_level: float = 0.0,
    defocus: float | tuple[float, float] = 0.0,
    NA: float | tuple[float, float] = None,
    wavelength: float = 1.0,
    scale: complex = 1.0 + 0j,
    normalize: bool = False,
    *,
    key: Key = None,
    **kwargs,
) -> ArrayLike:
    """Create a Gaussian-like initializer on the given `sampling` grid.

    Args:
        sampling: Target sampling grid.
        std: Standard deviation (or radius) of the Gaussian in pixels.
        noise_level: Uniform noise amplitude to add.
        defocus: Optional defocus phase to multiply with.

    Returns:
        Complex or real array shaped to `sampling` representing the Gaussian initializer.
    """

    std, defocus, key, shape, NA = _process_gaussian_args(sampling, radius, defocus, NA, wavelength, key)

    std = std / sampling.pixel_size

    x, y = np.meshgrid(
        np.arange(shape[0]) - shape[0] / 2,
        np.arange(shape[1]) - shape[1] / 2,
        indexing="ij",
    )
    coords = np.stack((x, y), axis=0)
    std = std[:, np.newaxis, np.newaxis]
    stretched_coords = coords / std
    stretched_r2 = jnp.sum(stretched_coords**2, axis=0)
    output = jnp.exp(-stretched_r2)
    noise = jax.random.uniform(key, shape=shape, dtype=jnp.float32)
    output += noise_level * noise

    if defocus != 0.0 and defocus != (0.0, 0.0) and (np.abs(defocus) > 1e-8).all():
        logging.debug(f"Applying defocus of {defocus} to Gaussian initializer")
        defocus = np.array(defocus)
        defocus = defocus / np.array(sampling.pixel_size)
        defocus = defocus[:, np.newaxis, np.newaxis]
        stretched_r2 = np.sum(np.sign(defocus) * (coords / defocus) ** 2, axis=0)
        exponential = phase_only_exp(2 * np.pi * stretched_r2)
        output = output * exponential

    return output * (1 + 0j) / jnp.linalg.norm(output) if normalize else output * scale


def _process_gaussian_args(
    sampling: SamplingGrid,
    radius: tuple | list | float,
    defocus: float | tuple[float, float],
    NA: float | tuple[float, float],
    wavelength: float,
    key: Key,
    std: float | tuple[float, float] = None,
) -> tuple[Array, Array, Key, tuple[int, int], Array]:
    shape = sampling.shape
    if radius is not None:
        std = radius
    if std is None:
        std = np.array(shape, dtype=float) / 8
    if isinstance(std, tuple | list | float):
        std = np.array(std, dtype=float)
    if key is None:
        key = jax.random.PRNGKey(42)
    if NA is not None and defocus == 0.0:
        NA = make_length_n(NA, 2)
        defocus = jnp.sqrt(2 * std * wavelength / NA)
        logging.info(f"Calculated defocus of {defocus} from NA of {NA} and std of {std}")
        defocus = tuple(defocus)

    std = np.array(std, dtype=float)
    std[std <= 0] = 1e8

    if not std.shape:
        std = np.array([std, std])
    if std.shape[0] == 1:
        std = np.array([std[0], std[0]])
    return std, defocus, key, shape, NA


@gin.configurable()
def speckle(
    sampling: SamplingGrid,
    std: float = None,
    radius: float = None,
    noise_level: float = 0.0,
    defocus: float | tuple[float, float] = 0.0,
    NA: float | tuple[float, float] = None,
    wavelength: float = 1.0,
    scale: complex = 1.0 + 0j,
    normalize: bool = False,
    *,
    key: Key = None,
    **kwargs,
) -> Float[Array, " m n"]:
    """Create a speckle-pattern initializer on a sampling grid.

    Generates a random speckle field by applying a random phase in the
    far-field plane and inverse-transforming, modulated by a Gaussian
    envelope.

    Args:
        sampling: Target sampling grid.
        std: Standard deviation of the real-space Gaussian envelope.
        radius: Alternative to ``std`` for specifying envelope size.
        noise_level: Amplitude of additive uniform noise.
        defocus: Defocus parameter (unused in speckle generation).
        NA: Numerical aperture controlling the far-field cutoff.
        wavelength: Wavelength for far-field coordinate computation.
        scale: Multiplicative scale factor.
        normalize: If True, normalize the output by its norm.
        key: JAX PRNG key for random phase generation.

    Returns:
        Complex speckle field array shaped to ``sampling``.
    """
    std, defocus, key, shape, NA = _process_gaussian_args(sampling, radius, defocus, NA, wavelength, key)
    defocus = np.array(defocus)
    defocus = defocus[:, np.newaxis, np.newaxis]
    noise_key, speckle_key = jax.random.split(key)
    coords = sampling.meshgrid
    std = std[:, np.newaxis, np.newaxis]
    stretched_coords = coords / std
    stretched_r2 = jnp.sum(stretched_coords**2, axis=0)

    f_coords = sampling.to_far_field().meshgrid
    NA = NA[:, np.newaxis, np.newaxis]
    stretched_f_coords = f_coords / NA
    stretched_f_r2 = jnp.sum(stretched_f_coords**2, axis=0)

    f_gaussian = jnp.exp(-stretched_f_r2)
    speckle_noise = jax.random.uniform(speckle_key, shape=shape, dtype=jnp.float32, maxval=2 * jnp.pi)
    f_speckle = f_gaussian * phase_only_exp(speckle_noise)
    real_speckle = ifft(f_speckle)

    speckle = jnp.exp(-stretched_r2) * real_speckle
    noise = jax.random.uniform(noise_key, shape=shape, dtype=jnp.float32)
    speckle += noise_level * noise

    return speckle / jnp.linalg.norm(speckle) if normalize else speckle * scale


@gin.configurable
def custom(
    sampling: SamplingGrid,
    weight: float = 1.0,
    std: float | tuple[float, float] | ArrayLike = None,
    radius: float = None,
    noise_level: float = 0.1,
    defocus: float = 0.0,
    scale: float = 1.0,
    *,
    key: Key,
    **kwargs,
) -> ArrayLike:
    """Compose a custom initializer by mixing a Gaussian probe with a shifted
    variant.

    Args:
        sampling: Target sampling grid.
        weight: Mixing weight between base and shifted probe.

    Returns:
        Complex array shaped to `sampling`.
    """

    gaussian_probe = gaussian(sampling, std, radius, noise_level, defocus, key=key, **kwargs) / 8
    shifted_gaussian = shift_with_interpolation(
        jnp.stack([gaussian_probe.real, gaussian_probe.imag], axis=0),
        jnp.array((20, 0)),
        gaussian_probe.shape,
    )
    output = (1 - weight) * gaussian_probe + weight * (shifted_gaussian[0] + 1j * shifted_gaussian[1])
    output = output / jnp.linalg.norm(output)
    output = output * scale
    return output


@gin.configurable()
def uniform(
    sampling: SamplingGrid, *args, scale: float = 1.0, normalize: bool = True, **kwargs
) -> Float[Array, "* m n"]:
    """Uniform initializer that returns a constant complex field on `sampling`.

    Args:
        sampling: Target sampling grid.
        normalize: If True, normalize the returned field.

    Returns:
        Complex array shaped to `sampling`.
    """

    shape = sampling.shape
    coefficients = np.ones(shape, dtype=np.complex64)
    coefficients = coefficients / jnp.linalg.norm(coefficients) if normalize else coefficients
    return coefficients * scale


@gin.configurable()
def random(sampling: SamplingGrid, *args, **kwargs) -> Float[Array, "* m n"]:
    """Random complex initializer on `sampling`.

    Args:
        sampling: Target sampling grid.

    Returns:
        Random complex array shaped to `sampling`.
    """

    shape = sampling.shape
    coefficients = np.random.random(shape) - 0.5 + 1j * (np.random.random(shape) - 0.5)
    return jnp.array(coefficients)


@gin.configurable()
def random_phase_mask(
    sampling: SamplingGrid,
    scale: float = 1.0,
    normalize: bool = False,
    *,
    key: Key,
    **kwargs,
) -> Float[Array, " m n"]:
    """Create a random phase mask initializer.

    Args:
        sampling: Target sampling grid.
        scale: Scaling factor for phase amplitude.
        normalize: Whether to normalize the output.

    Returns:
        Complex phase mask array shaped to `sampling`.
    """

    shape = sampling.shape
    random_phases = jax.random.uniform(key, shape=shape, minval=-jnp.pi, maxval=jnp.pi)
    phase_mask = phase_only_exp(scale * random_phases)
    if normalize:
        phase_mask = phase_mask / jnp.linalg.norm(phase_mask)
    return phase_mask


@gin.configurable()
def usaf_test_target(
    sampling: SamplingGrid,
    high_amplitude: float = 1.0,
    high_phase: float = 0.0,
    low_amplitude: float = 0.0,
    low_phase: float = 0.0,
    scale: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
    normalize: bool = False,
    svg_path: str = None,
    binarization_threshold: int = 200,
    svg_physical_size_m: float = 0.1,
) -> Complex[Array, " m n"]:
    """Create a USAF 1951 resolution test target initializer from SVG.

    Loads the USAF 1951 test target from an SVG file and scales it according to
    the sampling grid's pixel size and the scale parameter.

    The USAF 1951 standard dimensions:
    - Group 0, Element 1: 1.0 lp/mm (0.5 mm line width)
    - The SVG should be designed with these standard physical dimensions

    Args:
        sampling: Target sampling grid (pixel_size should be in meters).
        high_amplitude: Amplitude for the bars (test pattern).
        high_phase: Phase for the bars (radians).
        low_amplitude: Amplitude for the background.
        low_phase: Phase for the background (radians).
        scale: Overall scale factor for the pattern size (default=1.0 uses actual USAF dimensions).
               Use scale > 1 to make the pattern larger, < 1 to make it smaller.
        center: (x, y) center position of the pattern in physical units (meters).
        normalize: If True, normalize the output.
        svg_path: Path to the USAF SVG file. If None, uses default location.

    Returns:
        Complex array shaped to `sampling` representing the USAF test target.
    """
    import os

    from PIL import Image

    # Default path to USAF SVG
    if svg_path is None:
        svg_path = os.path.join(os.path.dirname(__file__), "..", "data", "example_images", "USAF-1951.svg")

    shape = sampling.shape
    pixel_size = sampling.pixel_size

    # Load and render SVG
    # First try using cairosvg if available, otherwise use PIL
    try:
        from io import BytesIO

        import cairosvg

        # Render SVG to PNG in memory with white background
        # (USAF pattern is transparent, needs background to be visible)
        png_data = cairosvg.svg2png(url=svg_path, background_color="white")
        img = Image.open(BytesIO(png_data)).convert("L")
    except ImportError:
        # Fallback: try PIL directly (requires Pillow with svg support)
        try:
            img = Image.open(svg_path).convert("L")
        except Exception as e:
            raise RuntimeError(
                f"Could not load SVG file. Please install cairosvg (pip install cairosvg) "
                f"or ensure SVG is pre-rendered as PNG. Error: {e}"
            ) from e

    # Convert to numpy array (0-255 grayscale)
    svg_array = np.array(img).astype(np.float32)

    # Normalize to binary mask
    # With white background, the pattern appears as dark (low values) on light (high values)
    # Threshold to detect the pattern: values significantly darker than white
    threshold = binarization_threshold
    binary_mask = svg_array < threshold

    svg_physical_size_m = svg_physical_size_m * scale

    # Calculate required output size in pixels based on pixel_size
    target_pixels_x = int(svg_physical_size_m / pixel_size[0])
    target_pixels_y = int(svg_physical_size_m / pixel_size[1])

    # Resize the binary mask to match target size
    binary_mask_resized = resize(
        binary_mask.astype(float),
        (target_pixels_x, target_pixels_y),
        order=0,  # Nearest neighbor for binary mask
        preserve_range=True,
        anti_aliasing=False,
    ).astype(bool)

    # Create output array and place the resized mask at the center
    high_value = high_amplitude * phase_only_exp(high_phase)
    low_value = low_amplitude * phase_only_exp(low_phase)
    pattern = jnp.ones(shape, dtype=jnp.complex64) * low_value

    # Calculate placement with centering
    center_offset_x = int(center[0] / pixel_size[0])
    center_offset_y = int(center[1] / pixel_size[1])

    start_x = (shape[0] - target_pixels_x) // 2 + center_offset_x
    start_y = (shape[1] - target_pixels_y) // 2 + center_offset_y

    # Clip to valid range
    end_x = min(start_x + target_pixels_x, shape[0])
    end_y = min(start_y + target_pixels_y, shape[1])
    start_x = max(0, start_x)
    start_y = max(0, start_y)

    # Extract the valid region of the mask
    mask_start_x = max(0, -start_x + (shape[0] - target_pixels_x) // 2 + center_offset_x)
    mask_start_y = max(0, -start_y + (shape[1] - target_pixels_y) // 2 + center_offset_y)
    mask_end_x = mask_start_x + (end_x - start_x)
    mask_end_y = mask_start_y + (end_y - start_y)

    # Create JAX array with the mask
    pattern_np = np.array(pattern)
    if mask_end_x > mask_start_x and mask_end_y > mask_start_y:
        pattern_np[start_x:end_x, start_y:end_y] = np.where(
            binary_mask_resized[mask_start_x:mask_end_x, mask_start_y:mask_end_y], high_value, low_value
        )

    pattern = jnp.array(pattern_np)

    if normalize:
        pattern = pattern / jnp.linalg.norm(pattern)

    return pattern


def separate_sampling_grid(func: Callable) -> Callable:
    """Decorator to adapt initializer functions to accept a chromatix
    SamplingGrid.

    Converts a function that takes ``(shape, pixel_size, ...)`` into one
    that takes ``(sampling: SamplingGrid, ...)`` by extracting ``shape``
    and ``pixel_size`` from the grid.

    Args:
        func: Initializer function with signature ``(shape, pixel_size, ...)``.

    Returns:
        Wrapped function accepting a :class:`SamplingGrid` as first argument.
    """

    @functools.wraps(func)
    def wrapper(sampling: SamplingGrid, *args, **kwargs):
        return func(sampling.shape, sampling.pixel_size, *args, **kwargs)

    return wrapper


# axicon_phase = separate_sampling_grid(chromatix.utils.axicon_phase)


@gin.configurable()
def siemens_star(
    sampling: SamplingGrid,
    num_spokes: int = 36,
    radius: float = None,
    anti_aliasing_factor: float | tuple[float, float] = 1.0,
    normalize: bool = True,
    min_value: float = 0.0,
    max_value: float = 1.0,
    **kwargs,
) -> Float[Array, " m n"]:
    """Generate a Siemens star test pattern on the given sampling grid.

    Uses chromatix to generate the spoke pattern and applies an anti-aliasing
    low-pass filter in the far field.

    Args:
        sampling: Target sampling grid.
        num_spokes: Number of spokes in the star.
        radius: Physical radius of the star; defaults to quarter of grid size.
        anti_aliasing_factor: Low-pass cutoff scaling for anti-aliasing.
        normalize: If True, normalize the output by its norm.
        min_value: Minimum intensity value in the pattern.
        max_value: Maximum intensity value in the pattern.

    Returns:
        Real array shaped to ``sampling`` representing the Siemens star.
    """
    if isinstance(anti_aliasing_factor, (float, int)):
        anti_aliasing_factor = np.array([anti_aliasing_factor, anti_aliasing_factor])

    from chromatix.data.data import siemens_star

    max_shape = min(sampling.shape)
    max_size = jnp.max(sampling.pixel_size)

    if radius is None:
        radius = max_shape / 4
    else:
        radius /= max_size

    image = siemens_star(max_shape, num_spokes, radius=radius) * (max_value - min_value) + min_value
    image = interpolate_grid_to_grid(
        image,
        SamplingGrid.from_tuples(image.shape, (max_size, max_size)),
        CoordinateSystem(),
        sampling,
        CoordinateSystem(),
        interpolation_mode="real_imaginary",
    )
    image = resize_to_match(image, sampling.shape)
    F_image = fft(image)
    f_sampling = sampling.to_far_field()
    xi_r_scaled = (f_sampling.xx / anti_aliasing_factor[0]) ** 2 + (f_sampling.yy / anti_aliasing_factor[1]) ** 2
    mask = xi_r_scaled < (0.5 / max_size) ** 2
    F_image = F_image * mask
    image = ifft(F_image)

    if normalize:
        image = image / jnp.linalg.norm(image)
    return image


@gin.configurable()
def support(
    sampling: SamplingGrid,
    size: ArrayLike = None,
    decay_length: float = 1.0,
    linear_power: int = 2,
    linear_size: ArrayLike = None,
) -> Float[Array, "n m"]:
    """Generate a support constraint weight map.

    Produces a smooth penalty that increases away from the center, useful as
    a regularization weight for sample boundaries.

    Args:
        sampling: Target sampling grid.
        size: Characteristic size of the support region (x, y).
        decay_length: Controls the smoothness of the transition at the boundary.
        linear_power: Exponent for the linear penalty term.
        linear_size: Size parameter for the linear growth region.

    Returns:
        Real weight array shaped to ``sampling``.
    """
    coordinates = sampling.meshgrid
    if size is None:
        size = (
            (jnp.max(coordinates[0]) - jnp.min(coordinates[0]) / 2),
            (jnp.max(coordinates[1]) - jnp.min(coordinates[1]) / 2),
        )
    if linear_size is None:
        linear_size = size

    size = jnp.array(size)
    linear_size = jnp.array(linear_size)

    rr = jnp.sqrt(jnp.sum((coordinates / size[:, jnp.newaxis, jnp.newaxis]) ** 2, axis=0))
    rr_linear = jnp.sqrt(jnp.sum((coordinates / linear_size[:, jnp.newaxis, jnp.newaxis]) ** 2, axis=0))

    support_weight = rr_linear**2 / (1 + jnp.exp(-(rr - 1) / decay_length**2))

    return support_weight
