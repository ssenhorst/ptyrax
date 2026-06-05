import glob
import itertools
import logging
import os
import pathlib
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from io import StringIO, TextIOWrapper
from typing import IO, Callable, Generator, Iterable, Iterator, Literal, Optional
from urllib.parse import urlparse

import gin
import h5py
import imageio.v3 as iio
import jax.numpy as jnp
import numpy as np
import pandas as pd
import requests
import scipy.io
import untangle
from jaxtyping import Array, ArrayLike, Float, Integer, Shaped
from matplotlib import pyplot as plt
from scipy.io.matlab import mat_struct
from scipy.linalg import solve
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from tqdm.auto import trange

from ptyrax.spatial import (
    R_y,
    matrix_to_six_dimensional_representation,
    meshgrid,
    shift_with_interpolation,
    six_dimensional_representation_to_matrix,
)
from ptyrax.utils import (
    load_hdf5,
    plot,
    save_hdf5,
)


def _download_file_from_url(url: str, output_path: str) -> None:
    """Download a remote file to ``output_path`` with a robust HTTPS
    fallback."""
    parsed = urlparse(url, scheme="https")
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must start with 'http:' or 'https:'")

    try:
        urllib.request.urlretrieve(url, output_path)  # noqa: S310
        return
    except Exception as urllib_exc:
        logging.warning("urllib download failed for %s, retrying with requests: %s", url, urllib_exc)

    try:
        with requests.get(url, stream=True, timeout=(10, 300)) as response:
            response.raise_for_status()
            with open(output_path, "wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_handle.write(chunk)
    except Exception as requests_exc:
        raise RuntimeError(f"Failed to download remote dataset from '{url}'") from requests_exc


class ImageDataset(ABC):
    @property
    @abstractmethod
    def images(self) -> Shaped[Array, "d m n"]:
        """Gets all images in the dataset as a single array.

        Returns:
            ArrayLike: The full (d, m, n) array of images.
        """
        pass

    @abstractmethod
    def to_gpu(self) -> None:
        pass

    def __array__(self) -> Shaped:
        """Allows the dataset to be used in arithmetic operations. This should
        usually just return one of the fields of the dataset. For example, if
        the dataset has a `diffraction_patterns` field, returning
        `self.diffraction_patterns` should be sufficient.

        Returns:
            Shaped: A `numpy` or `jax.numpy` array representing the dataset.
        """
        return self.images

    @property
    def n(self) -> int:
        """The number of positions in the dataset.

        Returns:
            int: The number of positions in the dataset
        """
        return len(self.images)

    @property
    def image_shape(self) -> int:
        """The shape of a single image in the dataset.

        Returns:
            tuple[int, int]: The shape of a single image.
        """
        return self.images.shape[1:]

    def __getitem__(self, item: slice) -> Shaped:
        """Specify behavior for how to slice the dataset.

        Args:
            item (slice): The slice to take

        Returns:
            ArrayLike: The sliced data
        """
        return self.images[item]

    def save(self, path: pathlib.Path) -> None:
        """A function to save the dataset to disk. By default, saves all fields
        of the dataclass to hdf5. If the dataset contains fields which are not
        supported by hdf5, this method should be overridden.

        Args:
            path (pathlib.Path): The path where to save the dataset.
        """
        try:
            save_hdf5(path, self.__dict__)
        except Exception as e:
            logging.error(
                f"Failed to save dataset to {path} in hdf5 format. Does your dataset contain unsupported hdf5 types?"
                f"If so, consider implementing a custom save method. Error: {e}"
            )
            raise e

    @classmethod
    def load(cls, path: pathlib.Path) -> "ImageDataset":
        """A function to load the dataset from disk. By default, loads all
        fields of the dataclass from hdf5. If the dataset contains fields which
        are not supported by hdf5, this method should be overridden.

        Args:
            path (pathlib.Path): The path where to load the dataset from.
        """
        try:
            data = load_hdf5(path)
            self = cls(**data)
        except Exception as e:
            logging.error(
                f"Failed to load dataset from {path} in hdf5 format. Does your dataset contain unsupported hdf5 types?"
                f"If so, consider implementing a custom load method. Error: {e}"
            )
            raise e
        return self

    __array_priority__ = 1000


class SimpleImageDataset(ImageDataset):
    """A simple implementation of ImageDataset that just wraps a single array
    of images.

    This can be used for simple cases where no additional metadata is
    needed.
    """

    def __init__(self, images: ArrayLike) -> None:
        self._images = np.array(images)

    @property
    def images(self) -> Shaped[Array, "d m n"]:
        return self._images

    def to_gpu(self) -> None:
        self._images = jnp.array(self._images)


@dataclass
class Ptychogram(ImageDataset):
    """Core dataset class for ptychographic reconstruction experiments.

    A Ptychogram holds diffraction patterns along with the full geometric
    metadata (scan positions, orientations, detector geometry, wavelength)
    needed for forward-model-based reconstruction.

    Coordinates follow the CXI convention: z is along the incoming beam,
    y is vertical. 2-D arrays are indexed ``(x, y)`` with ``'ij'``
    indexing.

    Args:
        diffraction_patterns: Measured intensity or amplitude patterns with
            shape ``(n, h, w)``.
        pixel_size: Detector pixel pitch as ``[dx, dy]``.
        sample_positions: Per-position sample translation vectors ``(n, 3)``
            in global coordinates.
        sample_orientations: Per-position 6-D orientation representations
            ``(n, 6)``.
        propagation_distance: Per-position sample-to-detector distance ``(n,)``.
        wavelength: One or more illumination wavelengths ``(m,)``.
        detector_positions: Per-position detector translation ``(n, 3)``.
        detector_orientations: Per-position detector orientation ``(n, 6)``.
        loaded_from: Human-readable string indicating the data source.
        diffraction_pattern_scale: Cumulative scaling factor applied to patterns.
        detector_darkframe: Detector dark-current image subtracted during
            preprocessing.
        mask: Optional boolean or float mask for invalid detector pixels.

    Example:
        >>> from ptyrax.dataset import from_hdf5
        >>> ptychogram = from_hdf5("data/lenspaper.hdf5")
        >>> ptychogram.n
        256
    """

    diffraction_patterns: Integer[Array, "n h w"] | Float[Array, "n h w"]
    pixel_size: Float[Array, "2"]
    sample_positions: Float[Array, "n 3"]
    sample_orientations: Float[Array, "n 6"]
    propagation_distance: Float[Array, "n"]
    wavelength: Float[Array, " m"]
    detector_positions: Float[Array, "n 3"]
    detector_orientations: Float[Array, "n 6"]
    loaded_from: str = "Not specified"
    diffraction_pattern_scale: float = 1.0
    detector_darkframe: np.ndarray = None
    mask: Optional[ArrayLike] = None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, item: slice) -> Integer[Array, " h w"] | Float[Array, " h w"]:
        return self.diffraction_patterns[item]

    def to_gpu(self) -> None:
        self.diffraction_patterns = jnp.array(self.diffraction_patterns)

    @property
    def n(self) -> int:
        return len(self.diffraction_patterns)

    @property
    def images(self) -> Integer[Array, " n h w"] | Float[Array, "n h w"]:
        return self.diffraction_patterns

    @classmethod
    def load_from(cls, path: pathlib.Path) -> "Ptychogram":
        if path.suffix in {".h5", ".hdf5"}:
            return cls.from_hdf5(path)
        elif path.suffix == ".cxi":
            return cls.from_cxi(path)
        else:
            raise ValueError(f"Can only load Ptychogram from hdf5 (*.h5, *.hdf5) or cxi (*.cxi). Got: {path.suffix}")

    def save(self, path: pathlib.Path) -> None:
        if path.suffix in {".h5", ".hdf5"}:
            self.to_hdf5(path)
        elif path.suffix == ".cxi":
            self.to_cxi(path)
        else:
            raise ValueError(f"Can only output Ptychogram to hdf5 (*.h5, *.hdf5) or cxi (*.cxi). Got: {path.suffix}")

    def __post_init__(self) -> None:
        """Validate and normalize fields after dataclass initialization.

        Ensures wavelength is a 1-D array, initializes the mask as a numpy array
        if provided, and creates a zero-valued darkframe if none was supplied.

        Raises:
            ValueError: If wavelength has more than one dimension.
        """
        self.wavelength = np.array(self.wavelength)
        if self.wavelength.shape == ():
            self.wavelength = self.wavelength[np.newaxis]
        if len(self.wavelength.shape) > 1:
            raise ValueError(
                "Wavelength format unknown."
                f"Wavelengths should be one-dimensional, but got shape {self.wavelength.shape}"
            )

        if self.mask is not None:
            self.mask = np.array(self.mask)

        if self.detector_darkframe is None:
            self.detector_darkframe = np.zeros_like(self.diffraction_patterns[0])

    def __plot__(self, *args, **kwargs) -> None:
        plot(np.max(self.diffraction_patterns, axis=0), *args, **kwargs)

    def __str__(self) -> str:
        def _print_attr(attr: str) -> str:  # noqa: ANN401
            value = getattr(self, attr)
            if isinstance(value, (np.ndarray)) or hasattr(value, "shape") and hasattr(value, "dtype"):
                return f"{attr}: array of shape {value.shape} and dtype {value.dtype}"
            else:
                return f"{attr}: {value}"

        attrs = "\n".join(_print_attr(attr) for attr in self.__dataclass_fields__.keys())
        return f"Ptychogram loaded from {self.loaded_from} with attributes:\n{attrs}"

    @property
    def pixel_number(self) -> tuple[int, ...]:
        """The number of pixels along each spatial dimension of a single
        diffraction pattern.

        Returns:
            tuple[int, ...]: A tuple ``(height, width)`` giving the detector pixel count.
        """
        return self.diffraction_patterns.shape[-2:]

    def to_hdf5(self, output_path: str) -> None:
        """Serialize the ptychogram to a flat HDF5 file.

        Each field is stored as a top-level dataset inside the file. The inverse
        operation is :func:`~ptyrax.dataset.from_hdf5`.

        Args:
            output_path: Filesystem path for the output ``.h5`` / ``.hdf5`` file.
        """
        with h5py.File(output_path, "w") as f:
            f.create_dataset("diffraction_patterns", data=self.diffraction_patterns)
            f.create_dataset("pixel_size", data=self.pixel_size)
            f.create_dataset("sample_positions", data=self.sample_positions)
            f.create_dataset("sample_orientations", data=self.sample_orientations)
            f.create_dataset("propagation_distance", data=self.propagation_distance)
            f.create_dataset("wavelength", data=self.wavelength)
            f.create_dataset("detector_positions", data=self.detector_positions)
            f.create_dataset("detector_orientations", data=self.detector_orientations)
            f.create_dataset("detector_darkframe", data=self.detector_darkframe)
            if self.mask is not None:
                f.create_dataset("mask", data=self.mask)

    def to_cxi(self, cxi_path: str) -> None:
        """Saves a Ptychogram object into a .cxi file (inverse of
        from_cxi())."""

        # Ensure directory exists
        os.makedirs(os.path.dirname(cxi_path), exist_ok=True)

        with h5py.File(cxi_path, "w") as f:
            entry = f.create_group("entry_1")

            # Data group
            data_grp = entry.create_group("data_1")
            data_grp.create_dataset("data", data=self.diffraction_patterns)
            data_grp.create_dataset("camera_pixel_size", data=self.pixel_size)

            # Instrument group
            instrument_grp = entry.create_group("instrument_1")

            # Detector
            det_grp = instrument_grp.create_group("detector_1")
            det_grp.create_dataset("x_pixel_size", data=self.pixel_size[0])
            det_grp.create_dataset("y_pixel_size", data=self.pixel_size[1])
            det_grp.create_dataset("distance", data=self.propagation_distance)
            det_grp.create_dataset("geometry_1/orientation", data=self.detector_orientations)
            det_grp.create_dataset("geometry_1/translation", data=self.detector_positions)
            det_grp.create_dataset("darkframe", data=self.detector_darkframe)

            # Source
            src_grp = instrument_grp.create_group("source_1")
            src_grp.create_dataset("wavelength", data=self.wavelength)

            # Sample
            sample_grp = entry.create_group("sample_1")
            geom_grp = sample_grp.create_group("geometry_1")
            geom_grp.create_dataset("translation", data=self.sample_positions)
            geom_grp.create_dataset("orientation", data=self.sample_orientations)

        logging.info(f"Ptychogram successfully saved to {cxi_path}")

    @gin.configurable
    def batch(
        self,
        batch_size: int = 1,
        shuffle_mode: str = "random",
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield batches of diffraction patterns and their indices.

        Args:
            batch_size (int): Number of samples per batch.
            shuffle_mode (str): One of 'random', 'by_distance', or 'clustered' to select batching order.

        Returns:
            Generator yielding a tuple of (indices, diffraction_pattern_batch).
        """
        num_batches = self.n // batch_size  # implicitly drops the last, incomplete batch
        remainder = self.n % batch_size
        if remainder > 0:
            logging.warning(
                "Dropping %d sample(s) from the last incomplete batch "
                "(%d samples, batch_size=%d). "
                "If shuffle is applied, different samples are dropped each epoch.",
                remainder,
                self.n,
                batch_size,
            )
        indices = np.arange(self.n)
        if shuffle_mode == "random":
            np.random.shuffle(indices)
        elif shuffle_mode == "by_distance":
            distance = np.linalg.norm(self.sample_positions, axis=-1)
            indices = indices[np.argsort(distance)]
        elif shuffle_mode == "clustered":
            from scipy.cluster.hierarchy import fcluster, linkage

            z = linkage(self.sample_positions, method="complete", metric="euclidean")
            clusters = fcluster(z, num_batches, criterion="distance")
            indices = indices[np.argsort(clusters)]

        for batch in trange(num_batches, desc="Batch", leave=False):
            current_index = indices[batch * batch_size : (batch + 1) * batch_size]
            diffraction_pattern_batch = self[current_index]
            yield np.array(current_index), diffraction_pattern_batch


def _convert_old_key_names(key_name: str) -> str:
    return "detector_darkframe" if key_name == "background" else key_name


def _build_ptychogram_from_flat_hdf5(
    data: dict,
    ptychogram_path: str,
    key_converter: Callable[[str], str] = _convert_old_key_names,
    convert_to_standard: bool = True,
) -> Ptychogram:
    """Convert a flattened HDF5-dict (as returned by `load_hdf5`) into a
    `Ptychogram`.

    If `convert_to_standard` is False, minimal mapping is performed and raw fields are used.
    """
    # Minimal mapping: prefer canonical names, otherwise fall back to legacy names
    ptychogram_required_keys = ("diffraction_patterns", "pixel_size", "sample_positions", "wavelength")
    ptychogram_optional_keys = (
        "mask",
        "sample_orientations",
        "detector_orientations",
        "detector_positions",
        "propagation_distance",
        "background",
        "detector_darkframe",
        "loaded_from",
    )
    ptychogram_keys = (*ptychogram_required_keys, *ptychogram_optional_keys)

    # Work on a shallow copy to avoid mutating caller data
    local = dict(data)
    n = len(local["diffraction_patterns"]) if "diffraction_patterns" in local else None

    if convert_to_standard:
        local = standardize_hdf5_shapes(n, local)
    # Build kwargs by preferring exact keys in the file, then falling back to the key_converter result
    kwargs = {}
    for k in ptychogram_keys:
        if k in local:
            kwargs[k] = local[k]
        else:
            alt = key_converter(k)
            kwargs[k] = local.get(alt, None)

    kwargs["loaded_from"] = ptychogram_path

    # Filter kwargs to only the fields accepted by the Ptychogram dataclass
    valid_fields = set(getattr(Ptychogram, "__dataclass_fields__", {}).keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

    if missing := [k for k in ptychogram_required_keys if k not in filtered_kwargs or filtered_kwargs.get(k) is None]:
        raise KeyError(
            "Could not initialise ptychogram from the hdf5 file. "
            f"Missing required keys: {missing}. Got: {list(local.keys())}"
        )

    try:
        return Ptychogram(**filtered_kwargs)
    except TypeError as e:
        raise KeyError(
            "Could not initialise ptychogram from the hdf5 file. "
            f"Constructor error: {e}. File keys: {list(local.keys())}"
        ) from e


def standardize_hdf5_shapes(n: int, local: dict) -> None:
    """Normalize raw HDF5 data fields to the canonical shapes expected by
    :py:class:`~ptyrax.dataset.Ptychogram`.

    Handles legacy key names, converts 2-D scan positions to 3-D, infers
    missing orientation and detector position fields from tilt angles and
    propagation distances, and renames the ``background`` key to
    ``detector_darkframe``.

    Args:
        n: Number of diffraction pattern frames in the dataset.
        local: Mutable dictionary of dataset fields loaded from HDF5. Modified
            in place and also returned.

    Returns:
        The updated dictionary with standardized shapes and keys.

    Raises:
        KeyError: If ``n`` is None (cannot determine frame count) or required
            position/distance fields are missing.
    """
    if n is None:
        raise KeyError("Cannot determine number of frames from 'diffraction_patterns'")

    if "tilt_angle" in local:
        rotation_matrix = Rotation.from_euler("xyz", local["tilt_angle"], degrees=True).as_matrix()
        local["sample_orientations"] = np.tile(matrix_to_six_dimensional_representation(rotation_matrix), (n, 1))

    if "sample_orientations" not in local.keys():
        local["sample_orientations"] = np.tile(matrix_to_six_dimensional_representation(np.eye(3)), reps=(n, 1))
        local["detector_orientations"] = np.tile(matrix_to_six_dimensional_representation(np.eye(3)), reps=(n, 1))

    if local.get("sample_positions") is not None and local["sample_positions"].shape[-1] == 2:
        logging.info("HDF5 has 2d scanning positions. Setting z = 0...")
        local["sample_positions"] = np.concatenate(
            [np.flip(local["sample_positions"], axis=-1), np.zeros((n, 1))], axis=-1
        )
        local["sample_positions"] = local["sample_positions"] * np.array([1.0, -1.0, 1.0])

    if "tilt_angle" in local:
        local["sample_positions"] = jnp.einsum(
            "ndi, ni -> nd",
            six_dimensional_representation_to_matrix(local["sample_orientations"]).transpose((0, 2, 1)),
            local["sample_positions"],
        )
        del local["tilt_angle"]

    if "detector_orientations" not in local.keys():
        detector_orientations = matrix_to_six_dimensional_representation(
            R_y(180)
            @ six_dimensional_representation_to_matrix(local["sample_orientations"])
            @ six_dimensional_representation_to_matrix(local["sample_orientations"])
        )
        local["detector_orientations"] = detector_orientations

        # If detector positions are missing but propagation distance is present, infer positions.
    if "detector_positions" not in local.keys():
        try:
            local = propagation_distance_to_full_position(n, local)
        except KeyError as e:
            raise KeyError(
                "Could not find 'detector_positions' or 'propagation_distance' in the hdf5 file. "
                "Either must be specified."
            ) from e

    if "background" in local:
        local["detector_darkframe"] = local["background"]
        del local["background"]
    else:
        local["detector_darkframe"] = np.zeros_like(local["diffraction_patterns"][0])
    return local


def propagation_distance_to_full_position(n: int, local: dict) -> None:
    """Convert a scalar or per-position propagation distance to 3-D detector
    positions.

    The propagation distance is interpreted as a displacement along the
    detector z-axis. If detector orientations are available they are used to
    transform the local z-offset into global coordinates.

    Args:
        n: Number of scan positions.
        local: Mutable dictionary containing at least ``propagation_distance``
            and optionally ``detector_orientations``. The computed
            ``detector_positions`` key is added in place.

    Returns:
        The updated dictionary with a ``detector_positions`` entry of shape
        ``(n, 3)``.

    Raises:
        KeyError: If ``propagation_distance`` is not present in *local*.
    """
    propagation = local["propagation_distance"]
    propagation = np.atleast_1d(propagation)
    if propagation.shape[0] == 1 and n > 1:
        propagation = np.tile(propagation, n)

    vec = np.stack((np.zeros_like(propagation), np.zeros_like(propagation), propagation), axis=-1)

    if "detector_orientations" in local:
        mats = six_dimensional_representation_to_matrix(local["detector_orientations"]).transpose((0, 2, 1))
        detector_positions = np.einsum("nij,nj->ni", mats, vec)
    else:
        detector_positions = vec

    local["detector_positions"] = detector_positions
    return local


# This was supposed to be a classmethod, but this does not play nice with gin.configurable...
@gin.configurable
def from_hdf5(
    ptychogram_path: str,
    key_converter: Callable[[str], str] = _convert_old_key_names,
    convert_to_standard: bool = True,
) -> Ptychogram:  # sourcery skip: remove-unnecessary-cast
    """Load a Ptychogram from an HDF5 or remote URL, autodetecting format.

    This function accepts local paths or HTTP(S) URLs. If given a URL the file is
    downloaded to a temporary file and inspected. The loader will dispatch to the
    appropriate sub-loader based on file attributes and keys.

    Args:
        ptychogram_path (str): Local filesystem path or HTTP(S) URL to the dataset.
        key_converter (callable): Function to normalize legacy key names.
        convert_to_standard (bool): If True, coerce legacy files to canonical fields.

    Returns:
        Ptychogram: The loaded ptychogram object.
    """

    def _is_url(p: str) -> bool:
        return isinstance(p, str) and (p.startswith("http://") or p.startswith("https://"))

    temp_path = None
    local_path = str(ptychogram_path)
    if _is_url(local_path):
        # Derive a safe suffix from the URL path extension, default to .h5
        url = urlparse(local_path, scheme="https")
        url_path = url.path
        ext = os.path.splitext(url_path)[1]
        suffix = ext if ext.lower() in (".h5", ".hdf5", ".cxi") else ".h5"
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            _download_file_from_url(local_path, temp_path)
            local_path = temp_path
        except Exception:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    # Inspect file to determine type
    try:
        with h5py.File(local_path, "r") as f:
            # check attributes first
            ds_type = None
            for attr_name in ("dataset_type", "type", "dataset_kind"):
                if attr_name in f.attrs:
                    ds_type = f.attrs[attr_name]
                    if isinstance(ds_type, bytes):
                        ds_type = ds_type.decode()
                    break

            keys = list(f.keys())

            # Heuristics: prefer explicit type attr, specific keys,
            # detect CXI format (contains 'entry_1' or CXI-style keys) and dispatch
            # to the CXI loader; otherwise fall back to ptychogram-like keys.
            # CXI files typically contain an 'entry_1' group with data_1 etc.
            if "entry_1" in keys or ("entry" in keys and "data_1" in f.get("entry", {})):
                return from_cxi(local_path)

            if "diffraction_patterns" in keys and ("sample_positions" in keys or "scan_pos" in keys):
                data = load_hdf5(local_path)
                return _build_ptychogram_from_flat_hdf5(
                    data, ptychogram_path, key_converter=key_converter, convert_to_standard=convert_to_standard
                )

            # If we didn't match any heuristics, raise a helpful error.
            raise ValueError(
                "Could not autodetect dataset type from HDF5 file. "
                "Expected attributes like 'dataset_type' or keys such as 'sample_positions'."
            )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as exc:
                logging.exception("Failed to remove temporary file %s: %s", temp_path, exc)


# This was supposed to be a classmethod, but this does not play nice with gin.configurable...
@gin.configurable()
def from_cxi(cxi_path: str, background_path: str = None) -> Ptychogram:
    """Loads a cxi file (as created by experiment_folder_to_cxi()) and returns
    a Ptychogram object."""
    import h5py

    with h5py.File(cxi_path, "r") as f:
        diffraction_patterns = f["entry_1/data_1/data"][()]
        pixel_size = np.array(
            [
                f["entry_1/instrument_1/detector_1/x_pixel_size"][()],
                f["entry_1/instrument_1/detector_1/y_pixel_size"][()],
            ]
        )
        sample_positions = f["entry_1/sample_1/geometry_1/translation"][()]
        scanning_orientations = f["entry_1/sample_1/geometry_1/orientation"][()]
        # orientation_matrix = six_dimensional_representation_to_matrix(scanning_orientations)
        # sample_positions = np.einsum('sjk, sk -> sj', orientation_matrix, sample_positions)
        # sample_positions = sample_positions[:, :2]  # We still only use the x and y coordinates

        detector_orientations = f["entry_1/instrument_1/detector_1/geometry_1/orientation"][()]
        detector_positions = f["entry_1/instrument_1/detector_1/geometry_1/translation"][()]
        propagation_distance = f["entry_1/instrument_1/detector_1/distance"][()]
        wavelength = f["entry_1/instrument_1/source_1/wavelength"][()]
        try:
            detector_darkframe = f["entry_1/instrument_1/detector_1/darkframe"][()]
        except KeyError:
            if background_path is not None:
                detector_darkframe = read_image(background_path)
                detector_darkframe = np.array(detector_darkframe, dtype=np.float32)
            else:
                detector_darkframe = np.zeros(diffraction_patterns.shape[1:])

    # Ensure propagation_distance is always a per-position array
    propagation_distance = np.atleast_1d(propagation_distance)
    n_positions = len(diffraction_patterns)
    if propagation_distance.shape[0] == 1 and n_positions > 1:
        propagation_distance = np.tile(propagation_distance, n_positions)

    return Ptychogram(
        diffraction_patterns=diffraction_patterns,
        pixel_size=pixel_size,
        sample_positions=sample_positions,
        propagation_distance=propagation_distance,
        wavelength=wavelength,
        sample_orientations=scanning_orientations,
        detector_positions=detector_positions,
        detector_orientations=detector_orientations,
        detector_darkframe=detector_darkframe,
        loaded_from=cxi_path,
    )


@gin.register
def fftshift_ptychogram(ptychogram: Ptychogram) -> Ptychogram:
    """Apply an FFT-shift to all diffraction patterns along the spatial axes.

    Swaps quadrants so that the zero-frequency component moves to the center
    of each pattern. This is required when raw detector data stores the DC
    component in the corner.

    Args:
        ptychogram: Input ptychogram with unshifted patterns.

    Returns:
        The same ptychogram with shifted ``diffraction_patterns``.
    """
    ptychogram.diffraction_patterns = np.fft.fftshift(ptychogram.diffraction_patterns, axes=(-2, -1))
    return ptychogram


@gin.register
def wavelength_units(ptychogram: Ptychogram) -> Ptychogram:
    """Rescale all length quantities so that the first wavelength becomes
    unity.

    Divides wavelengths, sample positions, detector positions, pixel sizes,
    and propagation distances by the first wavelength entry. Useful for
    working in dimensionless (wavelength-normalized) coordinates.

    Args:
        ptychogram: Input ptychogram with physical-unit lengths.

    Returns:
        The ptychogram with all lengths expressed in units of the first wavelength.
    """
    unit = ptychogram.wavelength[0]
    ptychogram.wavelength = ptychogram.wavelength / ptychogram.wavelength[0]
    ptychogram.sample_positions /= unit
    ptychogram.detector_positions /= unit
    ptychogram.pixel_size /= unit
    ptychogram.propagation_distance /= unit
    return ptychogram


@gin.register
def scale_length_unit(ptychogram: Ptychogram, scale: Float = 1.0) -> Ptychogram:
    """Multiply all length quantities by a constant factor.

    Applies the same scaling to wavelengths, sample positions, detector
    positions, pixel sizes, and propagation distances. Use this to convert
    between unit systems (e.g. metres to micrometres).

    Args:
        ptychogram: Input ptychogram.
        scale: Multiplicative factor applied to all length fields.

    Returns:
        The ptychogram with rescaled length quantities.
    """
    ptychogram.wavelength = ptychogram.wavelength * scale
    ptychogram.sample_positions *= scale
    ptychogram.detector_positions *= scale
    ptychogram.pixel_size *= scale
    ptychogram.propagation_distance *= scale
    return ptychogram


@gin.register
def shift_to_center_of_mass(ptychogram: Ptychogram, order: int = 2) -> Ptychogram:
    r"""Shift all diffraction patterns so the intensity center-of-mass is at the
    array center.

    Computes the intensity-weighted centroid of the mean diffraction pattern
    raised to the given power and applies a sub-pixel shift via interpolation.

    The center of mass is computed as:

    .. math::

        \mathbf{c} = \frac{\sum_{\mathbf{r}} \mathbf{r}\, I(\mathbf{r})^p}
                          {\sum_{\mathbf{r}} I(\mathbf{r})^p}

    where *p* is ``order``.

    Args:
        ptychogram: Input ptychogram.
        order: Power to which the mean pattern is raised before computing
            the centroid. Higher values emphasize the bright peak.

    Returns:
        The ptychogram with recentered diffraction patterns.
    """
    mean_diff_pat = np.mean(ptychogram.diffraction_patterns, axis=0)
    coords = meshgrid(ptychogram.pixel_number, pixel_size=(1.0, 1.0))
    center_of_mass = np.sum(coords * mean_diff_pat[np.newaxis] ** order, axis=(-2, -1)) / np.sum(mean_diff_pat**order)
    logging.info(f"Shifting ptychogram to center of mass: {center_of_mass}")
    shifted_diffraction_patterns = shift_with_interpolation(
        ptychogram.diffraction_patterns, np.flip(center_of_mass), ptychogram.pixel_number
    )
    ptychogram.diffraction_patterns = np.array(shifted_diffraction_patterns)
    return ptychogram


@gin.register
def non_negative(ptychogram: Ptychogram) -> Ptychogram:
    """Clamp negative pixel values in the diffraction patterns to zero.

    Negative values can appear after background subtraction or due to detector
    artifacts. This ensures all intensities are non-negative. Use with caution
    as this may introduce bias if negative values are significant. Really, it
    is better to fix the underlying issue in the loss function.

    Args:
        ptychogram: Input ptychogram.

    Returns:
        The ptychogram with all negative pixel values set to zero.
    """
    ptychogram.diffraction_patterns[ptychogram.diffraction_patterns < 0] = 0.0
    return ptychogram


@gin.register
def remove_zeros(ptychogram: Ptychogram) -> Ptychogram:
    """Replace exact-zero pixels with the minimum non-zero value.

    This avoids division-by-zero or log-of-zero issues during reconstruction
    while preserving the dynamic range of the data.

    Args:
        ptychogram: Input ptychogram.

    Returns:
        The ptychogram with zero pixels replaced by the per-pixel minimum
        of the non-zero values.
    """
    zeros = ptychogram.diffraction_patterns == 0
    non_zero_values = ptychogram.diffraction_patterns[~zeros]
    if non_zero_values.size == 0:
        raise ValueError(
            "All pixels in the diffraction patterns are zero. "
            "Cannot replace zeros — the dataset may be corrupted or empty."
        )
    ptychogram.diffraction_patterns[zeros] = np.min(non_zero_values, axis=0)
    return ptychogram


@gin.register
def exclude_positions_by_distance(
    ptychogram: Ptychogram,
    min_distance: float = -1.0,
    max_distance: float = 9999.0,
) -> Ptychogram:
    """Remove scan positions outside a distance range from the mean position.

    Filters out positions whose Euclidean distance from the centroid of all
    sample positions falls below ``min_distance`` or above ``max_distance``.
    Corresponding diffraction patterns and orientations are also removed.

    Args:
        ptychogram: Input ptychogram.
        min_distance: Minimum distance from the mean position to keep.
        max_distance: Maximum distance from the mean position to keep.

    Returns:
        The ptychogram with outlier positions removed.
    """
    mean_position = np.mean(ptychogram.sample_positions, axis=0)
    distances = np.linalg.norm(ptychogram.sample_positions - mean_position[np.newaxis], axis=-1)
    valid_positions = np.logical_and(distances > min_distance, distances < max_distance)
    ptychogram.sample_positions = ptychogram.sample_positions[valid_positions]
    ptychogram.sample_orientations = ptychogram.sample_orientations[valid_positions]
    ptychogram.diffraction_patterns = ptychogram.diffraction_patterns[valid_positions]
    ptychogram.detector_positions = ptychogram.detector_positions[valid_positions]
    ptychogram.detector_orientations = ptychogram.detector_orientations[valid_positions]
    return ptychogram


@gin.register
def normalize_by_max(ptychogram: Ptychogram, new_max: float = 1.0) -> Ptychogram:
    """Rescale diffraction patterns so the global maximum equals ``new_max``.

    Also rescales the detector darkframe and the stored
    ``diffraction_pattern_scale`` factor accordingly.

    Args:
        ptychogram: Input ptychogram.
        new_max: Target value for the brightest pixel across all patterns.

    Returns:
        The ptychogram with rescaled intensities.
    """
    scale = np.max(ptychogram.diffraction_patterns)
    ptychogram.diffraction_patterns = ptychogram.diffraction_patterns / scale * new_max
    ptychogram.detector_darkframe = ptychogram.detector_darkframe / scale * new_max
    ptychogram.diffraction_pattern_scale = ptychogram.diffraction_pattern_scale / scale * new_max
    return ptychogram


@gin.register
def normalize_by_mean(ptychogram: Ptychogram) -> Ptychogram:
    """Rescale diffraction patterns so their global mean value becomes unity.

    Also rescales the detector darkframe and stored scale factor.

    Args:
        ptychogram: Input ptychogram.

    Returns:
        The ptychogram with mean-normalized intensities.
    """
    scale = np.mean(ptychogram.diffraction_patterns)
    ptychogram.diffraction_patterns /= scale
    ptychogram.detector_darkframe /= scale
    ptychogram.diffraction_pattern_scale /= scale
    return ptychogram


@gin.register
def normalize_by_mean_intensity(ptychogram: Ptychogram) -> Ptychogram:
    """Rescale diffraction patterns by the mean of the per-pattern L2 norms.

    Computes the average Frobenius norm across all patterns and divides all
    intensities (and the darkframe/scale) by that value.

    Args:
        ptychogram: Input ptychogram.

    Returns:
        The ptychogram normalized by mean per-pattern intensity.
    """
    scale = np.mean(np.linalg.norm(ptychogram.diffraction_patterns, axis=(1, 2)))
    ptychogram.diffraction_patterns /= scale
    ptychogram.detector_darkframe /= scale
    ptychogram.diffraction_pattern_scale /= scale
    return ptychogram


@gin.configurable
def scale(ptychogram: Ptychogram, scale: float) -> Ptychogram:
    """Multiply all diffraction pattern values by a constant factor.

    Also updates the stored ``diffraction_pattern_scale``.

    Args:
        ptychogram: Input ptychogram.
        scale: Multiplicative scaling factor.

    Returns:
        The ptychogram with scaled diffraction patterns.
    """
    ptychogram.diffraction_patterns *= scale
    ptychogram.diffraction_pattern_scale *= scale
    return ptychogram


@gin.register
def intensity_to_amplitude(ptychogram: Ptychogram) -> Ptychogram:
    """Convert intensity-valued diffraction patterns to amplitude by taking the
    square root.

    Applies ``sqrt`` element-wise to both the diffraction patterns and the
    detector darkframe.

    Args:
        ptychogram: Input ptychogram with intensity values.

    Returns:
        The ptychogram with amplitude-valued patterns.
    """
    ptychogram.diffraction_patterns = np.sqrt(ptychogram.diffraction_patterns)
    ptychogram.detector_darkframe = np.sqrt(ptychogram.detector_darkframe)
    return ptychogram


def _apply_orientation_img(img: ArrayLike, orientation: int) -> Ptychogram:
    img_functions = {
        0: lambda x: x,
        1: lambda x: np.flip(x, -1),
        2: lambda x: np.flip(x, -2),
        3: lambda x: np.flip(np.flip(x, -2), -1),
        4: lambda x: np.transpose(x, (0, 2, 1)),
        5: lambda x: np.flip(np.transpose(x, (0, 2, 1)), -1),
        6: lambda x: np.flip(np.transpose(x, (0, 2, 1)), -2),
        7: lambda x: np.flip(np.flip(np.transpose(x, (0, 2, 1)), -1), -2),
    }
    return img_functions[orientation](img)


def _apply_orientation_coords(coords: ArrayLike, orientation: int) -> Ptychogram:
    coordinate_functions = {
        0: lambda x: x,
        1: lambda x: x,
        2: lambda x: x,
        3: lambda x: x,
        4: lambda x: np.flip(x),
        5: lambda x: np.flip(x),
        6: lambda x: np.flip(x),
        7: lambda x: np.flip(x),
    }
    return coordinate_functions[orientation](coords)


@gin.register
def apply_orientation(
    ptychogram: Ptychogram,
    orientation: Literal[0, 1, 2, 3, 4, 5, 6, 7] = 0,
    darkframe_orientation: Literal[0, 1, 2, 3, 4, 5, 6, 7] = None,
) -> Ptychogram:
    """Apply a geometric orientation transformation to diffraction patterns.

    The orientation code follows the convention:

    - 0: identity
    - 1: flip along y (last axis)
    - 2: flip along x (second-to-last axis)
    - 3: flip both axes
    - 4: transpose x and y
    - 5: transpose then flip y
    - 6: transpose then flip x
    - 7: transpose then flip both

    Also transforms the mask, darkframe, and pixel_size accordingly.

    Args:
        ptychogram: Input ptychogram.
        orientation: Integer code (0–7) specifying the desired transformation.
        darkframe_orientation: Separate orientation for the darkframe. Defaults
            to the same value as ``orientation``.

    Returns:
        The ptychogram with reoriented diffraction patterns.

    Raises:
        ValueError: If ``orientation`` is not in the range 0–7.
    """
    if darkframe_orientation is None:
        darkframe_orientation = orientation
    try:
        ptychogram.diffraction_patterns = _apply_orientation_img(ptychogram.diffraction_patterns, orientation)
        if ptychogram.mask is not None:
            # hack for position dimension
            ptychogram.mask = _apply_orientation_img(ptychogram.mask[np.newaxis, :, :], orientation)[0]
        ptychogram.detector_darkframe = _apply_orientation_img(
            ptychogram.detector_darkframe[np.newaxis], darkframe_orientation
        )[0]
        ptychogram.pixel_size = _apply_orientation_coords(ptychogram.pixel_size, orientation)
        return ptychogram
    except KeyError as e:
        raise ValueError(f"Invalid value for orientation. Expected an int between 0 and 7, got {orientation}") from e


@gin.register
def center_scan_positions(ptychogram: Ptychogram) -> Ptychogram:
    """Translate sample positions so their centroid is at the origin.

    Subtracts the mean of all sample positions, centering the scan around
    ``(0, 0, 0)``.

    Args:
        ptychogram: Input ptychogram.

    Returns:
        The ptychogram with zero-mean sample positions.
    """
    ptychogram.sample_positions -= np.mean(ptychogram.sample_positions, axis=0)
    return ptychogram


@gin.register
def scale_scan_positions(ptychogram: Ptychogram, scale: Float[Array, "3"]) -> Ptychogram:
    """Scale sample positions by a per-axis factor in the local sample frame.

    Positions are first transformed to the local frame defined by
    ``sample_orientations``, scaled by the given factor, and then transformed
    back to global coordinates.

    Args:
        ptychogram: Input ptychogram.
        scale: Array of length 3 giving the multiplicative scale factor for
            each local coordinate axis ``(x, y, z)``.

    Returns:
        The ptychogram with rescaled sample positions.
    """
    scale = np.array(scale)
    scale = np.atleast_1d(scale)
    local_frame_sample_positions = jnp.einsum(
        "ndi, ni -> nd",
        six_dimensional_representation_to_matrix(ptychogram.sample_orientations),
        ptychogram.sample_positions,
    )
    scaled_local_frame_sample_positions = local_frame_sample_positions * scale[np.newaxis, :]
    ptychogram.sample_positions = np.array(
        jnp.einsum(
            "ndi, ni -> nd",
            # transpose = inverse: From local to global
            six_dimensional_representation_to_matrix(ptychogram.sample_orientations).transpose((0, 2, 1)),
            scaled_local_frame_sample_positions,
        )
    )

    return ptychogram


@gin.register
def subtract_background(ptychogram: Ptychogram, background_path: str, orientation: int = 0) -> Ptychogram:
    """Subtract a background image loaded from file from all diffraction
    patterns.

    The background image is loaded from ``background_path``, optionally
    reoriented, and then subtracted element-wise from each pattern.

    Args:
        ptychogram: Input ptychogram.
        background_path: Path to the background image file (e.g. ``.png`` or
            ``.spe``).
        orientation: Orientation code (0–7) to apply to the background before
            subtraction.

    Returns:
        The ptychogram with background-subtracted patterns.
    """
    background, exp = read_image(background_path)
    background = _apply_orientation_img(background[np.newaxis], orientation)
    ptychogram.diffraction_patterns = np.array(ptychogram.diffraction_patterns - background)
    return ptychogram


def old_tud_key_converter(key: str) -> str:
    d = {
        "diffraction_patterns": "diff_pat",
        "pixel_size": "camera_pixel_size",
        "scanning_positions": "scan_pos",
        "propagation_distance": "prop_dist",
        "wavelength": "wavlen",
    }
    return d[key]


@gin.register
def flip_scanning_positions(ptychogram: Ptychogram) -> Ptychogram:
    """Reverse the order of coordinates within each sample position vector.

    Flips the last axis of ``sample_positions``, effectively swapping the
    x and z coordinates (with y in between).

    Args:
        ptychogram: Input ptychogram.

    Returns:
        The ptychogram with flipped sample position vectors.
    """
    ptychogram.sample_positions = np.flip(ptychogram.sample_positions, axis=-1)
    return ptychogram


@gin.register
def clip_low_intensity(ptychogram: Ptychogram, pixel_ratio: float = 0.90) -> Ptychogram:
    """Set pixels below a percentile threshold to zero.

    Computes the ``pixel_ratio * 100``-th percentile of all pixel values and
    zeros out any pixel at or below that threshold. This removes low-intensity
    background while preserving the bright signal.

    Args:
        ptychogram: Input ptychogram.
        pixel_ratio: Fraction (0–1) of pixels to zero out, specified as a
            percentile threshold.

    Returns:
        The ptychogram with low-intensity pixels clipped to zero.
    """
    # histogram, bins = np.histogram(ptychogram.diffraction_patterns.flatten(), bins=1000)
    # histogram = np.cumsum(histogram/np.sum(histogram))
    threshold = np.percentile(ptychogram.diffraction_patterns, pixel_ratio * 100)
    pats = np.array(ptychogram.diffraction_patterns)
    pats[ptychogram.diffraction_patterns <= threshold] = 0
    ptychogram.diffraction_patterns = pats
    return ptychogram


@gin.register
def subtract_low_intensity(ptychogram: Ptychogram, pixel_ratio: float = 0.90) -> Ptychogram:
    """Subtract a percentile-based threshold from all patterns and clamp to
    zero.

    Computes the ``pixel_ratio * 100``-th percentile across all pixel values,
    subtracts it uniformly, and sets any resulting negative values to zero.

    Args:
        ptychogram: Input ptychogram.
        pixel_ratio: Fraction (0–1) specifying the percentile used as the
            subtraction threshold.

    Returns:
        The ptychogram with the baseline subtracted.
    """
    threshold = np.percentile(ptychogram.diffraction_patterns, pixel_ratio * 100)
    pats = np.array(ptychogram.diffraction_patterns)
    pats = pats - threshold
    pats[pats <= 0.0] = 0.0
    ptychogram.diffraction_patterns = pats
    return ptychogram


@gin.register
def cut_center(ptychogram: Ptychogram, ratio: float = 0.5) -> Ptychogram:
    """Crop each diffraction pattern to a central sub-region.

    Keeps a fraction ``ratio`` of the total extent around the center along
    each spatial dimension.

    Args:
        ptychogram: Input ptychogram.
        ratio: Fraction (0–1) of the original extent to retain in each
            dimension.

    Returns:
        The ptychogram with cropped diffraction patterns.
    """
    center = (ptychogram.pixel_number[0] // 2, ptychogram.pixel_number[1] // 2)
    cut = (int(center[0] * ratio), int(center[1] * ratio))
    ptychogram.diffraction_patterns = ptychogram.diffraction_patterns[
        :, center[0] - cut[0] : center[0] + cut[0], center[1] - cut[1] : center[1] + cut[1]
    ]
    return ptychogram


@gin.register
def flip_scan_axis(ptychogram: Ptychogram, axis: int) -> Ptychogram:
    """Negate a component of all sample positions.

    Mirrors the scan pattern along the specified axis.

    Args:
        ptychogram: Input ptychogram.
        axis: Index of the axis to flip (0 for x, 1 for y, 2 for z).

    Returns:
        The ptychogram with the specified axis flipped.

    Raises:
        ValueError: If ``axis`` is not 0, 1, or 2.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    scale = np.ones(3)
    scale[axis] = -1
    ptychogram.sample_positions = ptychogram.sample_positions * scale
    return ptychogram


@gin.register
def flip_scan_x(ptychogram: Ptychogram) -> Ptychogram:
    """Negate the x-component of all sample positions."""
    return flip_scan_axis(ptychogram, axis=0)


@gin.register
def flip_scan_y(ptychogram: Ptychogram) -> Ptychogram:
    """Negate the y-component of all sample positions."""
    return flip_scan_axis(ptychogram, axis=1)


@gin.register
def scale_camera_distance(ptychogram: Ptychogram, scale: float) -> Ptychogram:
    """Multiply all detector positions by a constant factor.

    Effectively changes the sample-to-detector distance without modifying
    orientations.

    Args:
        ptychogram: Input ptychogram.
        scale: Multiplicative factor for detector positions.

    Returns:
        The ptychogram with rescaled detector positions.
    """
    ptychogram.detector_positions = ptychogram.detector_positions * scale
    return ptychogram


@gin.register
def scale_wavelength(ptychogram: Ptychogram, scale: float) -> Ptychogram:
    """Multiply the wavelength array by a constant factor.

    Args:
        ptychogram: Input ptychogram.
        scale: Multiplicative factor for wavelength.

    Returns:
        The ptychogram with rescaled wavelength.
    """
    ptychogram.wavelength = ptychogram.wavelength * scale
    return ptychogram


@gin.register
def make_multiwavelength(ptychogram: Ptychogram, wavelength_amount_factor: int = 1) -> Ptychogram:
    """Tile the wavelength array to simulate multi-wavelength illumination.

    Repeats the existing wavelength entries ``wavelength_amount_factor`` times.

    Args:
        ptychogram: Input ptychogram.
        wavelength_amount_factor: Number of times to tile the wavelength array.

    Returns:
        The ptychogram with an expanded wavelength array.
    """
    if not isinstance(ptychogram.wavelength, np.ndarray):
        ptychogram.wavelength = np.array(ptychogram.wavelength)
    ptychogram.wavelength = np.tile(ptychogram.wavelength, wavelength_amount_factor)
    return ptychogram


@gin.register
def mirror_coordinates(ptychogram: Ptychogram) -> Ptychogram:
    """Mirror the geometry about the y-z plane (negate x-coordinates).

    Flips the x-component of both sample and detector positions and adjusts
    the corresponding orientation representations to maintain consistency.

    Args:
        ptychogram: Input ptychogram.

    Returns:
        The ptychogram with mirrored coordinate geometry.
    """
    ptychogram.sample_positions = ptychogram.sample_positions * np.array([-1, 1, 1])
    ptychogram.detector_positions = ptychogram.detector_positions * np.array([-1, 1, 1])
    ptychogram.sample_orientations = ptychogram.sample_orientations * np.array([1, -1, -1, -1, 1, 1])
    ptychogram.detector_orientations = ptychogram.detector_orientations * np.array([-1, 1, 1, -1, 1, 1])
    return ptychogram


@gin.register
def make_constant_tilt_angle(
    ptychogram: Ptychogram,
    tilt_angle: float,
    detector_tilt_angle: float | None = None,
) -> Ptychogram:
    """Set a uniform sample tilt angle and recompute all geometry accordingly.

    Overrides the per-position sample orientations with a single rotation
    defined by ``tilt_angle`` (interpreted as rotation about y-axis in degrees).
    Sample positions are transformed to the new local frame, and detector
    orientations and positions are recomputed assuming specular geometry.

    Args:
        ptychogram: Input ptychogram.
        tilt_angle: Rotation angle about the y-axis in degrees.
        detector_tilt_angle: Optional separate tilt for the detector
            orientation. If ``None``, the detector orientation follows from
            the sample tilt via specular reflection.

    Returns:
        The ptychogram with updated orientations, positions, and propagation
        distances.
    """
    tilt_angle = jnp.array([0, tilt_angle, 0])

    def sample_orientation_from_tilt_angle(tilt_angle: float) -> np.ndarray:
        tilt_angle = np.array(tilt_angle)
        local_frame_sample_positions = jnp.einsum(
            "ndi, ni -> nd",
            six_dimensional_representation_to_matrix(ptychogram.sample_orientations),
            ptychogram.sample_positions,
        )
        rotation_matrix = Rotation.from_euler("xyz", tilt_angle, degrees=True).as_matrix()
        sample_orientations = np.tile(matrix_to_six_dimensional_representation(rotation_matrix), (ptychogram.n, 1))
        return sample_orientations, local_frame_sample_positions

    sample_orientations, local_frame_sample_positions = sample_orientation_from_tilt_angle(tilt_angle)
    sample_positions = jnp.einsum(
        "ndi, ni -> nd",
        # transpose = inverse: From local to global
        six_dimensional_representation_to_matrix(sample_orientations).transpose((0, 2, 1)),
        local_frame_sample_positions,
    )

    detector_orientations = matrix_to_six_dimensional_representation(
        R_y(180)
        @ six_dimensional_representation_to_matrix(sample_orientations)
        @ six_dimensional_representation_to_matrix(sample_orientations)
    )
    propagation_distances = np.linalg.norm(ptychogram.detector_positions, axis=-1)
    detector_positions = jnp.einsum(
        "ndi, ni -> nd",
        six_dimensional_representation_to_matrix(detector_orientations).transpose((0, 2, 1)),
        np.stack((0 * propagation_distances, 0 * propagation_distances, propagation_distances), axis=-1),
    )
    if detector_tilt_angle is not None:
        new_sample_orientations, _ = sample_orientation_from_tilt_angle(detector_tilt_angle)
        detector_orientations = matrix_to_six_dimensional_representation(
            R_y(180)
            @ six_dimensional_representation_to_matrix(new_sample_orientations)
            @ six_dimensional_representation_to_matrix(new_sample_orientations)
        )

    ptychogram.sample_positions = sample_positions
    ptychogram.sample_orientations = sample_orientations
    ptychogram.detector_positions = detector_positions
    ptychogram.detector_orientations = detector_orientations
    ptychogram.propagation_distance = propagation_distances  # Update per-position array

    return ptychogram


@gin.register
def set_constant_detector_positions(ptychogram: Ptychogram, constant_position: ArrayLike) -> Ptychogram:
    """Set all detector positions to a single constant value.

    Args:
        ptychogram: Input ptychogram.
        constant_position: 3-element array ``[x, y, z]`` specifying the
            detector position applied uniformly to all frames.

    Returns:
        The ptychogram with uniform detector positions.
    """
    constant_position = np.array(constant_position)
    new_positions = np.tile(constant_position[np.newaxis, :], (ptychogram.n, 1))
    ptychogram.detector_positions = new_positions
    return ptychogram


@gin.register
def set_constant_detector_orientations(ptychogram: Ptychogram, euler_angles: ArrayLike) -> Ptychogram:
    """Set all detector orientations to a single constant rotation.

    The rotation is specified as extrinsic Euler angles in the ``xyz``
    convention (in degrees) and is converted to a 6-D representation.

    Args:
        ptychogram: Input ptychogram.
        euler_angles: 3-element array ``[rx, ry, rz]`` of Euler angles in
            degrees.

    Returns:
        The ptychogram with uniform detector orientations.
    """
    rotation_matrix = Rotation.from_euler("xyz", euler_angles, degrees=True).as_matrix()
    constant_representation = matrix_to_six_dimensional_representation(rotation_matrix)
    new_representation = np.tile(constant_representation[np.newaxis, :], (ptychogram.n, 1))
    ptychogram.detector_orientations = new_representation
    return ptychogram


@gin.register
def add_constant_sample_shift(ptychogram: Ptychogram, constant_shift: ArrayLike) -> Ptychogram:
    """Add a constant offset to all sample orientation representations.

    Args:
        ptychogram: Input ptychogram.
        constant_shift: 6-element array added element-wise to each sample
            orientation vector.

    Returns:
        The ptychogram with shifted sample orientations.
    """
    constant_shift = np.array(constant_shift)
    ptychogram.sample_orientations = ptychogram.sample_orientations + constant_shift[np.newaxis]
    return ptychogram


@gin.register
def set_constant_sample_orientations(ptychogram: Ptychogram, euler_angles: ArrayLike) -> Ptychogram:
    """Set all sample orientations to a single constant rotation.

    The rotation is specified as extrinsic Euler angles in the ``xyz``
    convention (in degrees) and is converted to a 6-D representation.

    Args:
        ptychogram: Input ptychogram.
        euler_angles: 3-element array ``[rx, ry, rz]`` of Euler angles in
            degrees.

    Returns:
        The ptychogram with uniform sample orientations.
    """
    rotation_matrix = Rotation.from_euler("xyz", euler_angles, degrees=True).as_matrix()
    constant_representation = matrix_to_six_dimensional_representation(rotation_matrix)
    new_representation = np.tile(constant_representation[np.newaxis, :], (ptychogram.n, 1))
    ptychogram.sample_orientations = new_representation
    return ptychogram


@gin.register
def scale_diffraction_pattern_maximum(ptychogram: Ptychogram, maximum: float) -> Ptychogram:
    """Rescale diffraction patterns so the global maximum equals ``maximum``.

    First normalizes to [0, 1] by dividing by the current maximum, then
    multiplies by ``maximum``.

    Args:
        ptychogram: Input ptychogram.
        maximum: Desired maximum pixel value.

    Returns:
        The ptychogram with rescaled patterns.
    """
    ptychogram.diffraction_patterns /= np.max(ptychogram.diffraction_patterns)
    ptychogram.diffraction_patterns *= maximum
    return ptychogram


@gin.register
def amplitude_to_intensity(ptychogram: Ptychogram) -> Ptychogram:
    """Convert amplitude-valued diffraction patterns to intensity by squaring.

    Applies element-wise squaring to the diffraction patterns.

    Args:
        ptychogram: Input ptychogram with amplitude values.

    Returns:
        The ptychogram with intensity-valued patterns.
    """
    ptychogram.diffraction_patterns = np.square(ptychogram.diffraction_patterns)
    return ptychogram


@gin.register
def add_poisson_noise(
    ptychogram: Ptychogram,
    photons_per_count: float = None,
    total_photon_count: float = None,
    total_power: float = None,
    wavelength: float = None,
    exposure_time: float = None,
    diffraction_pattern_normalized: bool = True,
) -> Ptychogram:
    """Add Poisson-distributed shot noise to the diffraction patterns.

    Scales patterns to photon counts, draws Poisson samples, and rescales
    back. Exactly one of the following must be specified to determine the
    noise level:

    - ``photons_per_count``: direct conversion factor from pixel value to
      expected photon count.
    - ``total_photon_count``: total number of photons across all patterns.
    - ``total_power`` with ``wavelength`` and ``exposure_time``: computes
      total photon count from physical beam parameters.

    Args:
        ptychogram: Input ptychogram.
        photons_per_count: Conversion factor from detector counts to photons.
        total_photon_count: Total number of photons summed over all patterns.
        total_power: Beam power in watts (requires ``wavelength`` and
            ``exposure_time``).
        wavelength: Wavelength in metres (used with ``total_power``).
        exposure_time: Exposure time in seconds (used with ``total_power``).
        diffraction_pattern_normalized: If True, assume patterns are already
            normalized when computing scale from ``total_power``.

    Returns:
        The ptychogram with Poisson noise applied.

    Raises:
        ValueError: If none of the scaling parameters are specified.
    """
    if photons_per_count is not None:
        scale = photons_per_count
    elif total_photon_count is not None:
        scale = total_photon_count / np.sum(ptychogram.diffraction_patterns)
    elif total_power is not None and wavelength is not None and exposure_time is not None:
        # Calculate total photon count from power and wavelength
        h = 6.62607015e-34  # Planck's constant in J*s
        c = 299792458  # Speed of light in m/s
        energy_per_photon = h * c / wavelength  # Energy per photon in Joules
        total_photon_count = total_power * exposure_time / energy_per_photon
        scale = (
            total_photon_count / np.sum(ptychogram.diffraction_patterns)
            if diffraction_pattern_normalized
            else total_photon_count
        )
    else:
        raise ValueError(
            "Must specify one of photons_per_count, total_photon_count, or total_power with wavelength"
            "and exposure time."
        )

    scaled_patterns = ptychogram.diffraction_patterns * scale
    noisy_patterns = np.random.poisson(scaled_patterns).astype(np.float32)
    ptychogram.diffraction_patterns = noisy_patterns / scale
    return ptychogram


@gin.register
def add_gaussian_noise(ptychogram: Ptychogram, noise_mean: float, noise_variance: float) -> Ptychogram:
    """Add Gaussian noise to the diffraction patterns and clamp to non-
    negative.

    Draws from a normal distribution with the specified mean and standard
    deviation and adds it element-wise. Resulting negative values are clipped
    to zero.

    Args:
        ptychogram: Input ptychogram.
        noise_mean: Mean of the Gaussian noise distribution.
        noise_variance: Standard deviation of the Gaussian noise.

    Returns:
        The ptychogram with Gaussian noise added.
    """
    ptychogram.diffraction_patterns += np.random.normal(
        noise_mean, noise_variance, ptychogram.diffraction_patterns.shape
    )
    ptychogram.diffraction_patterns = np.clip(ptychogram.diffraction_patterns, a_min=0, a_max=None)
    return ptychogram


@gin.register
def quantize_diffraction_patterns(
    ptychogram: Ptychogram, dynamic_range_bits: int = 14, overexpose_fraction: float = 0.0
) -> Ptychogram:
    """Quantize diffraction patterns to simulate a finite bit-depth detector.

    Scales patterns to fill the dynamic range defined by ``dynamic_range_bits``,
    rounds to integer counts, and clips to the maximum value. An optional
    ``overexpose_fraction`` allows a fraction of the brightest pixels to
    saturate.

    Args:
        ptychogram: Input ptychogram.
        dynamic_range_bits: Number of bits defining the detector dynamic range
            (e.g. 14 gives a range of 0–16384).
        overexpose_fraction: Fraction of pixels allowed to saturate (0–1).

    Returns:
        The ptychogram with quantized diffraction patterns.
    """
    dynamic_range = 2**dynamic_range_bits
    scaled_patterns = (
        ptychogram.diffraction_patterns
        * dynamic_range
        / np.percentile(ptychogram.diffraction_patterns, (1 - overexpose_fraction) * 100)
    )
    ptychogram.diffraction_patterns = np.clip(np.round(scaled_patterns), a_min=0, a_max=dynamic_range)
    return ptychogram


default_column_matcher = {
    "x": "x \\[.*",
    "y": "y \\[.*",
    "z": "z \\[.*",
    "phi": "AOI \\[.*",
    "theta": "AZI \\[.*",
    "phi_prime": "CamRot.*",
}


def read_excel_columns(
    excel_path: str,
    column_matcher: dict[str, str] = None,
    data_filters: dict = None,
) -> dict:
    """Read columns from an Excel file by matching column headers with regex
    patterns.

    Each key in ``column_matcher`` maps a logical field name to a regex that is
    matched against the Excel column headers. Optional filters can restrict
    rows to specific values.

    Args:
        excel_path: Path to the ``.xlsx`` file.
        column_matcher: Dictionary mapping field names to regex patterns for
            matching column headers.
        data_filters: Optional dictionary of ``{column_name: value}`` pairs
            used to filter rows.

    Returns:
        Dictionary mapping field names to pandas Series of matched column data.

    Raises:
        ValueError: If a regex matches zero or more than one column.
    """
    if data_filters is None:
        data_filters = {}
    if column_matcher is None:
        column_matcher = default_column_matcher.copy()
    # Read the data from the Excel sheet using Pandas
    data = pd.read_excel(excel_path)
    column_data = {k: data.filter(regex=v).values for k, v in column_matcher.items()}

    for name, col in column_data.items():
        if col.shape[-1] == 0:
            raise ValueError(
                f"Could not find column {name} in the dataset excel file. The regex {column_matcher[name]} did not "
                f"match any of the columns {data.columns}."
            )
        elif col.shape[-1] > 1:
            raise ValueError(
                f"Found multiple columns to fit {name} in the dataset excel file. The regex {column_matcher[name]} "
                f"matched columns {data.filter(regex=column_matcher[name]).columns}."
            )
    column_data = {k: v.flatten() for k, v in column_data.items()}
    columns = pd.DataFrame(column_data)
    for f, val in data_filters.items():
        columns = columns[columns[f] == val]

    return columns.to_dict(orient="series")


def read_scan_pos_file(scan_pos_file: str, **kwargs) -> dict:
    """Dispatch scan position loading based on file extension.

    Supports ``.xlsx`` (Excel) and ``.mat`` (MATLAB) formats.

    Args:
        scan_pos_file: Path to the scan positions file.
        **kwargs: Additional keyword arguments passed to the format-specific
            reader.

    Returns:
        Dictionary with keys such as ``x``, ``y``, ``z``, ``phi``, ``theta``
        containing arrays of scan positions.
    """
    if scan_pos_file.endswith(".xlsx"):
        return read_excel_scan_pos(scan_pos_file, **kwargs)
    elif scan_pos_file.endswith(".mat"):
        return read_mat_scan_pos(scan_pos_file, **kwargs)


def read_excel_scan_pos(excel_path: str, precision: type = None) -> dict:
    """Read scan positions from an Excel file using fixed column indices.

    Expects columns at indices 3–7 to contain x, y, z, phi, and theta
    respectively.

    Args:
        excel_path: Path to the ``.xlsx`` file.
        precision: Optional numpy dtype to cast position values to.

    Returns:
        Dictionary with keys ``x``, ``y``, ``z``, ``phi``, ``theta``,
        ``phi_prime`` as numpy arrays.
    """
    # Read the data from the Excel sheet using Pandas
    data = pd.read_excel(excel_path)
    # Assuming the 4th column contains 'x' values and the 5th column contains 'y' values
    scan_pos = {
        "x": np.array(data.iloc[:, 3].values),
        "y": np.array(data.iloc[:, 4].values),
        "z": np.array(data.iloc[:, 5].values),
        "phi": np.array(data.iloc[:, 6].values),
        "theta": np.array(data.iloc[:, 7].values),
        "phi_prime": np.array(data.iloc[:, 6].values),
    }
    if precision is not None:
        for k, v in scan_pos.items():
            scan_pos[k] = v.astype(precision)

    return scan_pos


def read_mat_scan_pos(file_path: str, **kwargs) -> np.array:
    """Read scan positions from a MATLAB ``.mat`` metadata file.

    Extracts position arrays (x, y, z, phi, theta, phi_camera) from the
    ``metaData`` struct in the file.

    Args:
        file_path: Path to the ``.mat`` file.
        **kwargs: Unused; accepted for interface compatibility.

    Returns:
        Dictionary with keys ``x``, ``y``, ``z``, ``phi``, ``theta``,
        ``phi_camera`` as numpy arrays.
    """
    metadata = read_mat_metadata(file_path)

    x = np.array(metadata["Scanning_PositionsX"])
    y = np.array(metadata["Scanning_PositionsY"])

    def index_or_zero(dict: dict, key: str) -> np.array:
        try:
            return np.array(dict[key])
        except KeyError:
            return np.zeros_like(x)

    z = index_or_zero(metadata, "Scanning_PositionsZ")
    theta = index_or_zero(metadata, "Scanning_PositionsTheta")

    phi = index_or_zero(metadata, "Scanning_PositionsPhi")
    if "Angle_of_Incidence" in metadata.keys() and all(phi == 0):
        phi = np.ones_like(x) * metadata["Angle_of_Incidence"]

    phi_camera = index_or_zero(metadata, "Scanning_PositionsPhic")
    if all(phi_camera == 0):
        phi_camera = 2 * phi

    scan_pos = {
        "x": x,
        "y": y,
        "z": z,
        "phi": phi,
        "theta": theta,
        "phi_camera": phi_camera,
    }
    return scan_pos


def read_mat_metadata(file_path: str) -> dict:
    """Load and simplify the ``metaData`` struct from a MATLAB ``.mat`` file.

    Recursively converts MATLAB structs and structured arrays into plain
    Python dictionaries and lists.

    Args:
        file_path: Path to the ``.mat`` file containing a ``metaData`` variable.

    Returns:
        Nested dictionary representation of the MATLAB metadata struct.

    Raises:
        KeyError: If the file does not contain a ``metaData`` variable.
    """

    def simplify(obj: mat_struct | np.ndarray | Iterable) -> dict | list:
        if isinstance(obj, mat_struct):
            return {key: simplify(value) for key, value in obj.__dict__.items()}
        elif isinstance(obj, np.ndarray):
            if obj.dtype.names:  # Structured array
                return {name: simplify(obj[name]) for name in obj.dtype.names}
            elif obj.size == 1:
                return simplify(obj.item())
            else:
                return [simplify(el) for el in obj]
        else:
            return obj

    mat_data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)

    if "metaData" not in mat_data:
        raise KeyError("metaData struct not found in the .mat file.")

    meta_struct = mat_data["metaData"]
    simplified = simplify(meta_struct)

    return simplified


def sort_images_by_timestamp(image_paths: list[str]) -> list[str]:
    """Sort image file paths chronologically by embedded timestamp.

    Parses timestamps from filenames (format: ``YYYY Month DD HH_MM_SS``)
    and returns the paths sorted from earliest to latest. Handles Dutch
    month names.

    Args:
        image_paths: List of image file paths to sort.

    Returns:
        The same paths sorted by the timestamp extracted from filenames.

    Raises:
        ValueError: If a filename cannot be parsed into a valid timestamp.
    """
    dutch_to_english_months = {
        "januari": "January",
        "februari": "February",
        "maart": "March",
        "april": "April",
        "mei": "May",
        "juni": "June",
        "juli": "July",
        "augustus": "August",
        "september": "September",
        "oktober": "October",
        "november": "November",
        "december": "December",
    }

    # TODO remove duplicate
    def extract_timestamp(image_path: str) -> datetime:
        image_name = image_path.split("/")[-1]
        timestamp_info = image_name.split("-")[0]
        timestamp_info = timestamp_info.split(".")[0]
        for dutch_month, english_month in dutch_to_english_months.items():
            timestamp_info = timestamp_info.replace(dutch_month, english_month)  # unfortunate hack due to saving locale
        timestamp = datetime.strptime(timestamp_info, "%Y %B %d %H_%M_%S")
        return timestamp

    sorted_image_paths = sorted(image_paths, key=extract_timestamp)

    return sorted_image_paths


def read_png(file_path: str, precision: type = None) -> np.ndarray:
    """Read a PNG image file and return it as a numpy array.

    Args:
        file_path: Path to the ``.png`` file.
        precision: Numpy dtype to cast the image data to.

    Returns:
        Tuple of (image array, None). The second element is a placeholder
        for interface compatibility with other readers.
    """
    return iio.imread(file_path).astype(precision), None


def read_spe(file_path: str, precision: type = None) -> tuple[np.ndarray, float]:
    """Read a Princeton Instruments SPE file and return its image data.

    Extracts the last frame from the SPE file along with the exposure time
    metadata.

    Args:
        file_path: Path to the ``.spe`` file.
        precision: Numpy dtype to cast the image data to.

    Returns:
        Tuple of (image array, exposure_time_string).
    """
    image_file = SpeFile(file_path)

    data = np.asarray(*image_file.data[-1], dtype=precision)

    exposure_time_string = image_file.footer.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.ShutterTiming.ExposureTime.__dict__[  # noqa E501
        "cdata"
    ]

    # if precision is None:
    #     exposure_time = np.asarray(exposure_time_string, dtype=np.float_).item()
    # else:
    #     exposure_time = np.asarray(exposure_time_string, dtype=precision).item()

    return data, exposure_time_string


def read_image(file_path: str, precision: type = None) -> np.ndarray:
    """Read an image file, dispatching to the appropriate reader by extension.

    Supported formats: ``.png``, ``.spe``.

    Args:
        file_path: Path to the image file.
        precision: Numpy dtype to cast the image data to.

    Returns:
        Image as a numpy array.
    """
    read_mapping = {"png": read_png, "spe": read_spe}

    _, ext = os.path.splitext(file_path)
    image_or_tuple = read_mapping[ext[1:]](file_path, precision)
    if isinstance(image_or_tuple, tuple):
        return image_or_tuple[0]  # TODO fix read_spe such that it doesn't do this
    return image_or_tuple


def _raw_images(
    folder_path: str,
    precision: type = None,
    filter_fn: Callable = lambda x: True,
) -> Generator[tuple[np.ndarray, str], None, None]:
    image_ext = (".spe", "png")

    image_paths = [
        os.path.join(folder_path, image_name)
        for image_name in os.listdir(folder_path)
        if image_name.endswith(image_ext)
    ]

    image_paths = list(filter(filter_fn, image_paths))

    try:
        sorted_paths = sort_images_by_timestamp(image_paths)
    except ValueError:
        logging.warning("Could not sort images based on timestamp. Trying filenames...")
        sorted_paths = list(sorted(image_paths))

    for image_path in tqdm(sorted_paths):
        logging.info(image_path)
        image = read_image(image_path, precision)
        yield image, image_path


def load_raw_images(folder_path: str, precision: type = None, filter_fn: Callable = lambda x: True) -> np.ndarray:
    """Load all images from a folder into a single numpy array.

    Images are sorted by timestamp (or filename as fallback) before stacking.

    Args:
        folder_path: Path to the directory containing image files.
        precision: Numpy dtype to cast image data to.
        filter_fn: Callable that receives a file path and returns True to
            include it.

    Returns:
        Array of shape ``(n_images, height, width)`` containing all loaded
        images.
    """
    images = []
    images.extend(image for image, _ in _raw_images(folder_path, precision, filter_fn))
    # Warning: For large folders, this will consume a lot of memory!
    return np.array(images)


def experiment_folder_to_ptychogram_cxi(
    output_path: str,
    experiment_folder: str,
    extra_fields: dict,
    darkframe_folder: str | None = None,
    raw_folder: str | None = None,
    frames_folder: str | None = None,
    scan_pos_file: str | None = None,
    use_raw: bool = False,
    filename_filter_fn: Callable = lambda x: True,
    column_matcher: Callable = default_column_matcher,
    calibration: dict = {
        "PHI_C_OFF": 9.9146,  # deg
        "PHI_OFF": 5.837,  # deg
        "Z_BEAM_ALIGNED": 2770e-6,  # m
        "PHI_ZSTAGE": -(90 - 5.2),  # deg
    },  # TODO make calibration dataclass
) -> None:
    """Convert a raw experiment folder into a CXI-format HDF5 file.

    Reads raw images and scan position metadata from a TU Delft experiment
    folder layout, computes full 3-D sample/detector geometry using the
    provided calibration parameters, and writes a standards-compliant CXI
    file.

    Args:
        output_path: Output path for the ``.cxi`` file.
        experiment_folder: Root folder of the experiment data.
        extra_fields: Dictionary containing at least ``camera_pixel_size``,
            ``propagation_distance``, and ``wavelength``.
        darkframe_folder: Path to the darkframe images folder. Auto-detected
            if None.
        raw_folder: Path to raw image folder. Auto-detected if None.
        frames_folder: Path to processed frames folder. Auto-detected if None.
        scan_pos_file: Path to scan position file (``.xlsx`` or ``.mat``).
            Auto-detected if None.
        use_raw: If True, read from ``raw_folder`` instead of
            ``frames_folder``.
        filename_filter_fn: Filter function applied to image filenames.
        column_matcher: Regex patterns for matching scan position columns.
        calibration: Dictionary of instrument calibration parameters
            (offsets and geometry constants).
    """
    darkframe_folder, extra_fields, frames_folder, raw_folder, scan_pos_file = find_missing_folders(
        darkframe_folder, experiment_folder, extra_fields, frames_folder, raw_folder, scan_pos_file
    )

    def first_image_shape(scan_iterator: Iterator) -> tuple[int, "..."]:
        # Otherwise the actual iteration misses the first element
        scan_iterator, duplicate_iterator = itertools.tee(scan_iterator)
        for scan_data in duplicate_iterator:
            image, path = scan_data[-1]
            return image.shape

    def append_to_dataset(dataset: h5py.Dataset, data: np.ndarray) -> None:
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = data

    required_fields = ("camera_pixel_size", "propagation_distance", "wavelength")
    for required_field in required_fields:
        if required_field not in extra_fields.keys():
            raise KeyError(
                f"Missing required field {required_field} in extra_fields. extra_fields should always contain "
                f"{required_fields}."
            )

    scan_pos = read_scan_pos_file(scan_pos_file, column_matcher=column_matcher)

    with h5py.File(output_path, "w") as f:
        # Set CXI version
        f.create_dataset("cxi_version", data=130)

        # Entry group
        entry = f.create_group("entry_1")

        # Instrument group
        instrument = entry.create_group("instrument_1")
        source = instrument.create_group("source_1")
        wavelength = np.array(extra_fields["wavelength"])

        source.create_dataset("wavelength", data=wavelength)
        source.create_dataset("spectrum", data=np.array(extra_fields.get("spectrum", np.ones_like(wavelength))))

        detector = instrument.create_group("detector_1")
        detector.create_dataset("distance", data=extra_fields["propagation_distance"])
        detector.create_dataset("x_pixel_size", data=extra_fields.get("camera_pixel_size")[0])
        detector.create_dataset("y_pixel_size", data=extra_fields.get("camera_pixel_size")[1])
        darkframes = load_raw_images(darkframe_folder)
        detector.create_dataset("darkframes", data=darkframes)

        detector_geometry = detector.create_group("geometry_1")
        # Direction cosines of the basis vectors of the local coordinate system in the global coordinate system
        # Order is x'.x, x'.y, x'.z, y'.x, y'.y, y'.z
        detector_orientation_dataset = detector_geometry.create_dataset(
            "orientation", shape=(0, 6), maxshape=(None, 6), dtype=np.float32
        )
        detector_translation_dataset = detector_geometry.create_dataset(
            "translation", shape=(0, 3), maxshape=(None, 3), dtype=np.float32
        )
        detector_geometry.create_dataset("propagation_distance", data=extra_fields["propagation_distance"])

        data_group = entry.create_group("data_1")
        image_shape = darkframes.shape[-2:]
        image_dataset = data_group.create_dataset(
            "data", shape=(0, *image_shape), maxshape=(None, *image_shape), dtype=np.float32
        )
        # Metadata
        data_group.create_dataset("camera_pixel_size", data=extra_fields["camera_pixel_size"])
        # image_dataset.attrs["axes"] = "translation:y:x"

        # Sample geometry
        sample = entry.create_group("sample_1")
        sample_geometry = sample.create_group("geometry_1")
        # Translation respect to the *GLOBAL* coordinate system (defined by sample_orientation)
        sample_translation_dataset = sample_geometry.create_dataset(
            "translation", shape=(0, 3), maxshape=(None, 3), dtype=np.float32
        )

        # Direction cosines of the basis vectors of the local coordinate system in the global coordinate system
        # Order is x'.x, x'.y, x'.z, y'.x, y'.y, y'.z
        sample_orientation_dataset = sample_geometry.create_dataset(
            "orientation", shape=(0, 6), maxshape=(None, 6), dtype=np.float32
        )
        sample_geometry.create_dataset(
            "azimuth_rotation_origin", data=extra_fields.get("theta_rotation_origin", np.array([0, 0, 0]))
        )

        # Remake the generator
        image_gen = (
            _raw_images(raw_folder, filter_fn=filename_filter_fn)
            if use_raw
            else _raw_images(frames_folder, filter_fn=filename_filter_fn)
        )
        scan_iterator = zip(
            scan_pos["x"],
            scan_pos["y"],
            scan_pos["z"],
            scan_pos["phi"],
            scan_pos["theta"],
            scan_pos["phi_camera"],
            image_gen,
        )
        # Data acquisition
        for scan_data in tqdm(scan_iterator):
            x_smaract, y_smaract, z_smaract, phi, theta, phi_camera, (image, path) = scan_data
            phi = 90 - (phi - calibration["PHI_OFF"])
            # phi_c = phi_camera - calibration['PHI_C_OFF']
            # We ignore z for now (i.e. we define the origin where the beam hits the sample for given z)
            # Calculate orientation matrix: matrix which when applied to the local coordinate system x'
            # transforms to (1, 0, 0), etc.
            # sample_orientation = R_z(theta) @ R_y(phi) #TODO include theta into the axes!
            sample_orientation = R_y(phi)
            stage_orientation = R_y(phi)
            append_to_dataset(
                sample_orientation_dataset, matrix_to_six_dimensional_representation(sample_orientation)
            )  # Flatten to store as 1D array

            # x, and y stages are rotated by phi only, the azimuthal rotation comes last in the stack.
            # (assuming rotation axis at x=0, y=0)
            # Transpose: Local coordinates to global
            z_stage_axis = R_y(calibration["PHI_ZSTAGE"])[:, 2]
            # Absolute z translation: motion from state where rotation axes of sample and camera are aligned
            z_stage_translation = z_smaract * 1e-6 * z_stage_axis
            sample_phi_axis_z_stage_translation = z_stage_translation - calibration["Z_BEAM_ALIGNED"] * z_stage_axis
            z_axis = np.array([0.0, 0.0, 1.0])
            # x-coordinate: translation vector to the origin of phi rotation, y-coordinate (always 0),
            # z-coordinate: change in origin coordinates (along z only, because this is the incoming beam direction.
            beam_sample_intersection = solve(
                np.stack([stage_orientation[0], stage_orientation[1], z_axis], axis=-1),
                sample_phi_axis_z_stage_translation,
            )

            sample_translation = stage_orientation.T @ np.array(
                (x_smaract * 1e-6 + beam_sample_intersection[0], -y_smaract * 1e-6, 0)
            )  # Negative sign: positive y of stages is pointing downwards
            if not np.isclose((sample_orientation @ sample_translation)[2], 0):
                raise ValueError(
                    "In sample coordinates, there should be no shift in z (this would cause beam defocus!)"
                )
            append_to_dataset(sample_translation_dataset, sample_translation)
            # phi_prime is approximately 2 * phi for specular reflection
            detector_orientation = R_y(180) @ R_y(phi) @ R_y(phi)
            append_to_dataset(
                detector_orientation_dataset, matrix_to_six_dimensional_representation(detector_orientation)
            )  # Flatten to store as 1D array
            # First move to origin of Z_BEAM_ALIGNED coordinates, then to origin where beam hits the sample
            origin_translation_camera_rotation_axis = (
                calibration["Z_BEAM_ALIGNED"] * z_stage_axis + beam_sample_intersection[2]
            )
            detector_translation = detector_orientation.T @ np.array((0, 0, extra_fields["propagation_distance"]))
            detector_translation += origin_translation_camera_rotation_axis
            append_to_dataset(detector_translation_dataset, detector_translation)

            append_to_dataset(image_dataset, image)
            logging.info(f"Time: {os.path.basename(path)[9:15]} X: {x_smaract}, Y: {y_smaract}")

    logging.info(f"Converted experiment folder to cxi to {output_path}")


def find_missing_folders(
    darkframe_folder: str | None,
    experiment_folder: str | None,
    extra_fields: dict,
    frames_folder: str | None,
    raw_folder: str | None,
    scan_pos_file: str | None,
) -> tuple[str, dict, str, str, str]:
    """Auto-detect missing folder and file paths from the experiment layout.

    Inspects the ``experiment_folder`` for standard sub-directories
    (``frames``, ``RAW``, ``darkframe``) and scan position files (``.xlsx``,
    ``.mat``) to fill in any arguments left as None.

    Args:
        darkframe_folder: Explicit darkframe folder path or None to auto-detect.
        experiment_folder: Root folder of the experiment.
        extra_fields: Dictionary of extra metadata fields (passed through).
        frames_folder: Explicit frames folder or None.
        raw_folder: Explicit raw folder or None.
        scan_pos_file: Explicit scan position file or None.

    Returns:
        Tuple of ``(darkframe_folder, extra_fields, frames_folder,
        raw_folder, scan_pos_file)`` with resolved paths.

    Raises:
        NotADirectoryError: If darkframe folder cannot be found.
    """
    if os.path.exists(os.path.join(experiment_folder, "ExperimentalData")):
        experiment_folder = os.path.join(experiment_folder, "ExperimentalData")

    if extra_fields is None:
        extra_fields = {}

    scan_pos_file = _resolve_scan_pos_file(experiment_folder, scan_pos_file)
    frames_folder = _resolve_frames_folder(experiment_folder, frames_folder, raw_folder)
    raw_folder = _resolve_raw_folder(experiment_folder, raw_folder)

    if darkframe_folder is None:
        for name in ("darkframe", "darkFrame", "DarkFrame", "DarkFrame"):
            candidate = os.path.join(experiment_folder, name)
            if os.path.exists(candidate):
                darkframe_folder = candidate
                break

    if darkframe_folder is None:
        raise NotADirectoryError(
            "Could not find the folder specifying the darkframes automatically. Please specify manually."
        )

    return darkframe_folder, extra_fields, frames_folder, raw_folder, scan_pos_file


def _resolve_scan_pos_file(experiment_folder: str, scan_pos_file: str | None) -> str:
    """Determine scan_pos_file if not provided explicitly."""
    if scan_pos_file is not None:
        return scan_pos_file

    excel_or_matlab_files = glob.glob(os.path.join(experiment_folder, "*.xlsx")) + glob.glob(
        os.path.join(experiment_folder, "*.mat")
    )
    if len(excel_or_matlab_files) > 1:
        logging.warning(
            f"Multiple files specifying scanning position data found {excel_or_matlab_files}. "
            f"Using {excel_or_matlab_files[0]} for scanning positions."
        )
    if not excel_or_matlab_files:
        raise FileNotFoundError(
            "Could not find the excel or matlab file specifying the scanning positions. "
            "Please provide this as an argument."
        )
    return excel_or_matlab_files[0]


def _resolve_frames_folder(
    experiment_folder: str,
    frames_folder: str | None,
    raw_folder: str | None,
) -> str:
    """Determine frames_folder if not provided explicitly."""
    if frames_folder is not None:
        return frames_folder

    for name in ("frames",):
        candidate = os.path.join(experiment_folder, name)
        if os.path.exists(candidate):
            return candidate

    # Preserve original error message semantics, but avoid using a None path in exists()
    missing = raw_folder if raw_folder is not None else "<unknown>"
    raise NotADirectoryError(
        f"The specified experiment folder does not contain a subfolder ({missing}). Could not process the folder"
    )


def _resolve_raw_folder(experiment_folder: str, raw_folder: str | None) -> str:
    """Determine raw_folder if not provided explicitly."""
    if raw_folder is not None:
        return raw_folder

    for name in ("RAW", "raw", "raw_frames"):
        candidate = os.path.join(experiment_folder, name)
        if os.path.exists(candidate):
            return candidate

    missing = raw_folder if raw_folder is not None else "<unknown>"
    raise NotADirectoryError(
        f"The specified experiment folder does not contain a subfolder ({missing}). Could not process the folder"
    )


def experiment_folder_to_ptychogram_hdf5(
    output_ptychogram_hdf5: str,
    experiment_folder: str,
    extra_fields: dict,
    darkframe_folder: str | None = None,
    raw_folder: str | None = None,
    scan_pos_file: str | None = None,
    filter: Callable | None = None,
) -> None:
    """Convert a raw experiment folder into a flat ptychogram HDF5 file.

    A simpler alternative to :func:`experiment_folder_to_ptychogram_cxi` that
    writes a flat HDF5 file loadable by :func:`from_hdf5`.

    Args:
        output_ptychogram_hdf5: Output path for the ``.hdf5`` file.
        experiment_folder: Root folder of the experiment data.
        extra_fields: Dictionary containing at least ``tilt_angle``,
            ``camera_pixel_size``, ``propagation_distance``, and
            ``wavelength``.
        darkframe_folder: Path to darkframe folder. Auto-detected if None.
        raw_folder: Path to raw image folder. Auto-detected if None.
        scan_pos_file: Path to scan position file. Auto-detected if None.
        filter: Optional filename filter function.

    Raises:
        KeyError: If required fields are missing from ``extra_fields``.
    """
    darkframe_folder, extra_fields, frames_folder, raw_folder, scan_pos_file = find_missing_folders(
        darkframe_folder,
        experiment_folder,
        extra_fields,
        frames_folder=None,
        raw_folder=raw_folder,
        scan_pos_file=scan_pos_file,
    )

    # subfolders = glob.glob(os.path.join(raw_folder, '*/'))
    # if len(subfolders) > 0 and not HDR:
    #     logging.warning(f'Found multiple subfolders in the raw data folder (likely due to multiple exposures),
    #     but HDR is disabled. Using the first subfolder: {subfolders[0]}')
    #     raw_folder = subfolders[0]

    scan_pos = read_excel_scan_pos(scan_pos_file)
    images = load_raw_images(raw_folder, filter_fn=filter)
    darkframes = load_raw_images(darkframe_folder)
    background = np.mean(darkframes, axis=0)
    mask = np.ones_like(images[0])
    required_fields = (
        "tilt_angle",
        "camera_pixel_size",
        "propagation_distance",
        "wavelength",
    )
    for required_field in required_fields:
        if required_field not in extra_fields.keys():
            raise KeyError(
                f"Missing required field {required_field} in extra_fields. extra_fields should always contain "
                f"{required_fields}."
            )

    data = {
        "diffraction_pattern": images,
        "scan_pos": scan_pos,
        "tilt_angle": extra_fields["tilt_angle"],
        "camera_pixel_size": extra_fields["camera_pixel_size"],
        "background": background,
        "mask": mask,
        "propagation_distance": extra_fields["propagation_distance"],
        "wavelength": extra_fields["wavelength"],
    }
    save_all_in_hdf5(data.values(), data.keys(), output_ptychogram_hdf5)


def save_all_in_hdf5(data_list: list, dataset_name_list: list["str"], hdf5_path: str) -> None:
    """Save multiple arrays to a single HDF5 file.

    Each array is stored as a top-level dataset with the corresponding name.
    Existing datasets with the same name are overwritten.

    Args:
        data_list: Iterable of arrays to save.
        dataset_name_list: Iterable of dataset names (one per array).
        hdf5_path: Output HDF5 file path.
    """
    with h5py.File(hdf5_path, "w") as hdf5_handle:
        for data, dataset_name in zip(data_list, dataset_name_list):
            if dataset_name in hdf5_handle:
                del hdf5_handle[dataset_name]

            hdf5_handle[dataset_name] = data

            logging.info(dataset_name + " saved...")


def plot_dataset_dynamic_range(dataset: str, output_path: str, dpi: int = 200) -> None:
    """Plot and save a pixel intensity histogram for the dataset.

    Reads diffraction patterns from an HDF5 file, computes a histogram of all
    pixel values, and saves the resulting plot to ``output_path``.

    Args:
        dataset: Path to the HDF5 file containing a ``diff_pat`` dataset.
        output_path: Directory where the ``pixel_histogram.png`` file will
            be saved.
        dpi: Resolution of the saved figure in dots per inch.
    """
    with h5py.File(dataset) as file:
        diff_pat = file["diff_pat"][:]
        # background = file['background'][:]
    counts, bins, patch = plt.hist(diff_pat.flatten(), bins=2000, log=True)
    threshold = 0.5
    threshold_bin = np.percentile(counts, threshold * 100)
    plt.axvline(threshold_bin, label="50%")
    plt.title("Pixel histogram")
    plt.xlabel("Intensity (AU)")
    plt.ylabel("Counts")
    plt.savefig(os.path.join(output_path, "pixel_histogram.png"), dpi=dpi)
    plt.close()


class SpeFile:
    """Reader for Princeton Instruments SPE v3.x format files.

    Parses the binary header, XML footer, and raw image data from SPE files
    produced by Princeton Instruments cameras (e.g. via LightField software).
    Supports multiple frames and regions of interest.

    Args:
        filepath: Path to the ``.spe`` file to read.

    Attributes:
        filepath: Path to the source file.
        header_version: SPE format version number from the binary header.
        nframes: Number of frames in the file.
        footer: Parsed XML footer as an ``untangle.Element`` tree.
        dtype: Numpy dtype of the stored image data.
        xdim: List of x-dimensions for each region of interest.
        ydim: List of y-dimensions for each region of interest.
        roi: List of region-of-interest metadata elements.
        nroi: Number of regions of interest.
        wavelength: Wavelength calibration array (if available).
        data: Nested list ``[frame][roi]`` of image arrays.
        metadata: Per-frame metadata array or None.
        metanames: List of metadata field names or None.

    Raises:
        ValueError: If ``filepath`` is not a string or the SPE version is < 3.0.
    """

    def __init__(self, filepath: str) -> None:
        if filepath is None:
            logging.warning(
                "Deprecation Warning: construct via gui has been deprecated in this module. "
                "Use load() in spe2py instead."
            )
            return
        if not isinstance(filepath, str):
            raise ValueError("Filepath must be a single string")
        self.filepath = filepath

        with open(self.filepath) as file:
            self.header_version = read_at(file, 1992, 3, np.float32)[0]
            if self.header_version < 3.0:
                raise ValueError(f"This version of spe2py cannot load filetype SPE v. {self.header_version:.1f}")

            self.nframes = read_at(file, 1446, 2, np.uint16)[0]

            self.footer = self._read_footer(file)
            self.dtype = self._get_dtype(file)

            # Note: these methods depend on self.footer
            self.xdim, self.ydim = self._get_dims()
            self.roi, self.nroi = self._get_roi_info()
            self.wavelength = self._get_wavelength()

            self.xcoord, self.ycoord = self._get_coords()

            self.data, self.metadata, self.metanames = self._read_data(file)
        file.close()

    @staticmethod
    def _read_footer(file: TextIOWrapper) -> untangle.Element:
        """Loads and parses the source file's xml footer metadata to an
        'untangle' object."""
        footer_pos = read_at(file, 678, 8, np.uint64)[0]

        file.seek(footer_pos)
        xmltext = file.read()

        parser = untangle.make_parser()
        sax_handler = untangle.Handler()
        parser.setContentHandler(sax_handler)

        parser.parse(StringIO(xmltext))

        loaded_footer = sax_handler.root

        return loaded_footer

    @staticmethod
    def _get_dtype(file: TextIOWrapper) -> type:
        """Returns the numpy data type used to encode the image data by reading
        the numerical code in the binary header.

        Reference: Princeton Instruments File Specification pdf
        """
        dtype_code = read_at(file, 108, 2, np.uint16)[0]

        if dtype_code == 0:
            dtype = np.float32
        elif dtype_code == 1:
            dtype = np.int32
        elif dtype_code == 2:
            dtype = np.int16
        elif dtype_code == 3:
            dtype = np.uint16
        elif dtype_code == 8:
            dtype = np.uint32
        else:
            raise ValueError(f"Unrecognized data type code: {dtype_code}. Value should be one of {0, 1, 2, 3, 8}")

        return dtype

    def _get_meta_dtype(self) -> tuple[list[type], list[str]]:
        meta_types = []
        meta_names = []
        prev_item = None
        for item in dir(self.footer.SpeFormat.MetaFormat.MetaBlock):
            if item == "TimeStamp" and prev_item != "TimeStamp":  # Specify ExposureStarted vs. ExposureEnded
                for element in self.footer.SpeFormat.MetaFormat.MetaBlock.TimeStamp:
                    meta_names.append(element["event"])
                    meta_types.append(element["type"])
                prev_item = "TimeStamp"
            elif item == "GateTracking" and prev_item != "GateTracking":  # Specify Delay vs. Width
                for element in self.footer.SpeFormat.MetaFormat.MetaBlock.GateTracking:
                    meta_names.append(element["component"])
                    meta_types.append(element["type"])
                prev_item = "GateTracking"
            elif prev_item != item:  # All other metablock names only have one possible value
                meta_names.append(item)
                meta_types.append(getattr(self.footer.SpeFormat.MetaFormat.MetaBlock, item)["type"])
                prev_item = item

        for index, type_str in enumerate(meta_types):
            meta_types[index] = np.int64 if type_str == "Int64" else np.float64
        return meta_types, meta_names

    def _get_roi_info(self) -> tuple[list, int]:
        """Returns region of interest attributes and numbers of regions of
        interest."""
        try:
            camerasettings = self.footer.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera
            regionofinterest = camerasettings.ReadoutControl.RegionsOfInterest.CustomRegions.RegionOfInterest
        except AttributeError:
            # print("XML Footer was not loaded prior to calling _get_roi_info")
            raise

        if isinstance(regionofinterest, list):
            nroi = len(regionofinterest)
            roi = regionofinterest
        else:
            nroi = 1
            roi = [regionofinterest]  # cast element to list for consistency

        return roi, nroi

    def _get_wavelength(self) -> np.ndarray:
        """Returns wavelength-to-pixel map as stored in XML footer."""
        try:
            wavelength_string = StringIO(self.footer.SpeFormat.Calibrations.WavelengthMapping.Wavelength.cdata)
        except (AttributeError, IndexError):
            # print("XML Footer was not loaded prior to calling _get_wavelength or \n"
            #       "XML Footer does not contain Wavelength Mapping information")
            return
        wavelength = np.loadtxt(wavelength_string, delimiter=",")

        return wavelength

    def _get_dims(self) -> tuple[list[int], list[int]]:
        """Returns the x and y dimensions for each region as stored in the XML
        footer."""
        xdim = [int(block["width"]) for block in self.footer.SpeFormat.DataFormat.DataBlock.DataBlock]
        ydim = [int(block["height"]) for block in self.footer.SpeFormat.DataFormat.DataBlock.DataBlock]

        return xdim, ydim

    def _get_coords(self) -> tuple[list[list[int]], list[list[int]]]:
        """Returns x and y pixel coordinates.

        Used in cases where xdim and ydim do not reflect image
        dimensions (e.g. files containing frames with multiple regions
        of interest)
        """
        xcoord = [[] for _ in range(self.nroi)]
        ycoord = [[] for _ in range(self.nroi)]

        for roi_ind in range(self.nroi):
            working_roi = self.roi[roi_ind]
            ystart = int(working_roi["y"])
            ybinning = int(working_roi["yBinning"])
            yheight = int(working_roi["height"])
            ycoord[roi_ind] = range(ystart, (ystart + yheight), ybinning)

        for roi_ind in range(self.nroi):
            working_roi = self.roi[roi_ind]
            xstart = int(working_roi["x"])
            xbinning = int(working_roi["xBinning"])
            xwidth = int(working_roi["width"])
            xcoord[roi_ind] = range(xstart, (xstart + xwidth), xbinning)

        return xcoord, ycoord

    def _read_data(self, file: TextIOWrapper) -> tuple[list, np.ndarray | None, list[str] | None]:
        """Loads raw image data into an nframes X nroi list of arrays."""
        file.seek(4100)

        frame_stride = int(self.footer.SpeFormat.DataFormat.DataBlock["stride"])
        frame_size = int(self.footer.SpeFormat.DataFormat.DataBlock["size"])
        metadata_size = frame_stride - frame_size
        if metadata_size != 0:
            metadata_dtypes, metadata_names = self._get_meta_dtype()
            metadata = np.zeros((self.nframes, len(metadata_dtypes)))
        else:
            metadata_dtypes, metadata_names = None, None
            metadata = None

        data = [[0 for _ in range(self.nroi)] for _ in range(self.nframes)]
        for frame in range(self.nframes):
            for region in range(self.nroi):
                if self.nroi > 1:
                    data_xdim = len(self.xcoord[region])
                    data_ydim = len(self.ycoord[region])
                else:
                    data_xdim = np.asarray(self.xdim[region], np.uint32)
                    data_ydim = np.asarray(self.ydim[region], np.uint32)
                data[frame][region] = np.fromfile(file, self.dtype, data_xdim * data_ydim).reshape(data_ydim, data_xdim)
            if metadata_dtypes is not None:
                for meta_block in range(len(metadata_dtypes)):
                    metadata[frame, meta_block] = np.fromfile(file, dtype=metadata_dtypes[meta_block], count=1)

        return data, metadata, metadata_names


def load_spe_from_files(filepaths: list[str]) -> list[SpeFile] | None:
    """Allows user to load multiple files at once.

    Each file is stored as an SpeFile object in the list batch.
    """
    if filepaths is None:
        logging.warning(
            "Deprecation Warning: load via gui has been deprecated in this module.Use load() in spe2py instead."
        )
        return
    batch = [[] for _ in range(len(filepaths))]
    for file in range(len(filepaths)):
        batch[file] = SpeFile(filepaths[file])
    return_type = "list of SpeFile objects"
    if len(batch) == 1:
        batch = batch[0]
        return_type = "SpeFile object"
    logging.info("Successfully loaded %i file(s) in a %s" % (len(filepaths), return_type))
    return batch


def read_at(file: IO, pos: int, size: int, ntype: type) -> np.ndarray:
    """Reads SPE source file at specific byte position.

    Adapted from
    https://scipy.github.io/old-wiki/pages/Cookbook/Reading_SPE_files.html
    """
    file.seek(pos)
    return np.fromfile(file, ntype, size)
