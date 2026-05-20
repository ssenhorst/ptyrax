from __future__ import annotations

import logging
from typing import Dict

import jax.numpy as jnp
import numpy as np

from ptyrax.field import CoherentField
from ptyrax.spatial import (
    CoordinateSystem,
    Rotation,
    SamplingGrid,
    matrix_to_six_dimensional_representation,
)


def state_get_with_candidates(state: Dict[str, np.ndarray], candidates: list[str]) -> np.ndarray:
    """Retrieve a value from a state dict by trying multiple candidate keys.

    Args:
        state: Dictionary mapping HDF5-like paths to arrays.
        candidates: Ordered list of keys to try.

    Returns:
        The array found under the first matching key.

    Raises:
        KeyError: If none of the candidate keys exist in ``state``.
    """
    for key in candidates:
        if key in state:
            return state[key]
    raise KeyError(f"None of the expected keys were found: {candidates}")


def state_pick_index(arr: np.ndarray, index: int, min_ndim: int = 2) -> np.ndarray:
    """Select along axis 0 if the array has at least ``min_ndim`` dimensions.

    Args:
        arr: Input array.
        index: Index to select along axis 0.
        min_ndim: Minimum dimensionality required to perform indexing.

    Returns:
        Indexed slice if applicable, otherwise ``arr`` unchanged.
    """
    arr = np.asarray(arr)
    return arr[index] if arr.ndim >= min_ndim and arr.shape[0] > index else arr


def state_get_optional(state: Dict[str, np.ndarray], candidates: list[str]) -> np.ndarray | None:
    """Retrieve a value from state by candidate keys, returning None if
    missing.

    Args:
        state: Dictionary mapping HDF5-like paths to arrays.
        candidates: Ordered list of keys to try.

    Returns:
        The first matched array, or None if no candidate is found.
    """
    return next((state[key] for key in candidates if key in state), None)


def state_get_array(
    state: Dict[str, np.ndarray],
    candidates: list[str],
    *,
    index: int = 0,
    min_ndim: int = 2,
    target_ndim: int = 1,
    min_size: int | None = None,
    label: str = "",
) -> np.ndarray:
    """Retrieve, index, reshape and validate an array from a state dict.

    Combines the common pattern of ``state_get_with_candidates`` followed by
    optional index selection, reshape, and size validation.

    Args:
        state: Dictionary mapping HDF5-like paths to arrays.
        candidates: Ordered list of keys to try.
        index: Index to select along axis 0 when applicable.
        min_ndim: Minimum dimensionality to trigger index selection.
        target_ndim: If ``1``, the result is reshaped to 1-D via ``.reshape(-1)``.
        min_size: Minimum number of elements required. Raises ``ValueError``
            if the resulting array has fewer elements.
        label: Descriptive label used in error messages.

    Returns:
        The processed NumPy array.

    Raises:
        KeyError: If none of the candidate keys are found.
        ValueError: If the array has fewer than ``min_size`` elements.
    """
    arr = np.asarray(state_get_with_candidates(state, candidates))
    if arr.ndim >= min_ndim:
        arr = state_pick_index(arr, index=index, min_ndim=min_ndim)
    arr = np.asarray(arr)
    if target_ndim == 1:
        arr = arr.reshape(-1)
    if min_size is not None and arr.size < min_size:
        raise ValueError(f"{label} must have at least {min_size} components, got {arr.size}.")
    return arr


def coherent_field_from_hdf5_state(
    state: Dict[str, np.ndarray],
    *,
    index: int = 0,
    probe_path_prefix: str = "illumination/_probe",
) -> CoherentField:
    """Build a CoherentField from a loaded reconstruction state dictionary."""

    pref = probe_path_prefix.strip("/")
    ppref = f"{pref}/" if pref else ""

    probe_data = np.asarray(
        state_get_with_candidates(
            state,
            [
                f"{ppref}_data",
                f"{ppref}data",
            ],
        )
    )
    if probe_data.ndim >= 4:
        probe_data = state_pick_index(probe_data, index=index, min_ndim=4)
    if probe_data.ndim == 2:
        probe_data = probe_data[..., np.newaxis]

    wavelength_arr = np.asarray(state_get_with_candidates(state, [f"{ppref}wavelength"]))
    wavelength_arr = wavelength_arr.reshape(-1)
    if wavelength_arr.size == 0:
        raise ValueError(f"Missing wavelength under '{ppref}wavelength'.\"")
    clamped_index = min(index, wavelength_arr.size - 1)
    if clamped_index != index:
        logging.warning(
            "Requested wavelength index %d exceeds available size %d; clamping to %d.",
            index,
            wavelength_arr.size,
            clamped_index,
        )
    wavelength = float(wavelength_arr[clamped_index])

    propagation_direction = np.asarray(state_get_with_candidates(state, [f"{ppref}propagation_direction"]))
    if propagation_direction.ndim >= 2:
        propagation_direction = state_pick_index(propagation_direction, index=index, min_ndim=2)
    propagation_direction = np.asarray(propagation_direction).reshape(-1)
    if propagation_direction.size < 3:
        raise ValueError(f"Propagation direction under '{ppref}propagation_direction' must have 3 components.")

    rotation_candidates = [
        f"{ppref}coordinate_system/parameters/rotation/_representation_6d",
        f"{ppref}coordinate_system/rotation/_representation_6d",
        f"{ppref}coordinates/parameters/rotation/_representation_6d",
        f"{ppref}coordinates/rotation/_representation_6d",
    ]
    try:
        rotation_6d = np.asarray(state_get_with_candidates(state, rotation_candidates))
        if rotation_6d.ndim >= 2:
            rotation_6d = state_pick_index(rotation_6d, index=index, min_ndim=2)
        rotation_6d = np.asarray(rotation_6d).reshape(-1)
        if rotation_6d.size < 6:
            raise ValueError("Rotation representation must have 6 components.")
    except KeyError:
        logging.warning(
            "Rotation not found in state for prefix '%s'; falling back to identity rotation.",
            ppref,
        )
        rotation_6d = np.asarray(matrix_to_six_dimensional_representation(jnp.eye(3))).reshape(-1)

    translation_candidates = [
        f"{ppref}coordinate_system/parameters/_translation",
        f"{ppref}coordinate_system/_translation",
        f"{ppref}coordinate_system/translation",
        f"{ppref}coordinates/parameters/_translation",
        f"{ppref}coordinates/_translation",
        f"{ppref}coordinates/translation",
    ]
    try:
        translation = np.asarray(state_get_with_candidates(state, translation_candidates))
        if translation.ndim >= 2:
            translation = state_pick_index(translation, index=index, min_ndim=2)
        translation = np.asarray(translation).reshape(-1)
    except KeyError:
        logging.warning(
            "Translation not found in state for prefix '%s'; falling back to zero translation.",
            ppref,
        )
        translation = np.zeros(3, dtype=float)

    pixel_size = np.asarray(state_get_with_candidates(state, [f"{ppref}sampling/pixel_size"]))
    pixel_size = np.asarray(pixel_size).reshape(-1)
    if pixel_size.size == 1:
        pixel_size = np.repeat(pixel_size, 2)
    if pixel_size.size < 2:
        raise ValueError(f"Pixel size under '{ppref}sampling/pixel_size' must provide at least one value.")

    if probe_data.ndim < 3:
        raise ValueError(f"Probe data under '{ppref}_data' must have at least 3 dimensions (m, n, d).")
    field_shape = tuple(int(v) for v in probe_data.shape[-3:-1])

    probe_sampling = SamplingGrid.from_tuples(field_shape, tuple(float(v) for v in pixel_size[:2]))
    probe_coordinate_system = CoordinateSystem(
        rotation=Rotation(jnp.asarray(rotation_6d)),
        translation=jnp.asarray(translation[:3]),
    )

    return CoherentField(
        data=jnp.asarray(probe_data),
        wavelength=jnp.asarray(wavelength),
        sampling=probe_sampling,
        coordinate_system=probe_coordinate_system,
        propagation_direction=jnp.asarray(propagation_direction[:3]),
    )


def post_interaction_field_from_hdf5_state(
    state: Dict[str, np.ndarray],
    *,
    index: int = 0,
    interaction_path_prefix: str = "interaction",
    probe_path_prefix: str = "illumination/_probe",
) -> CoherentField:
    """Build a post-interaction field using the same reflection convention as
    FresnelReflection."""

    probe = coherent_field_from_hdf5_state(state, index=index, probe_path_prefix=probe_path_prefix)

    pref = interaction_path_prefix.strip("/")
    ipref = f"{pref}/" if pref else ""

    interaction_rotation = np.asarray(
        state_get_with_candidates(
            state,
            [
                f"{ipref}coordinates/parameters/rotation/_representation_6d",
                f"{ipref}coordinates/rotation/_representation_6d",
            ],
        )
    )
    if interaction_rotation.ndim >= 2:
        interaction_rotation = state_pick_index(interaction_rotation, index=index, min_ndim=2)

    sample_rotation = Rotation(jnp.asarray(np.asarray(interaction_rotation).reshape(-1)[:6]))
    incident_direction_sample = sample_rotation.as_matrix() @ probe.propagation_direction
    reflected_direction = incident_direction_sample * jnp.array([1.0, 1.0, -1.0])

    forward_pixel_size = np.asarray(
        state_get_with_candidates(
            state,
            [
                f"{ipref}forward_sampling/pixel_size",
                f"{ipref}sampling/pixel_size",
                f"{probe_path_prefix.strip('/')}/sampling/pixel_size",
            ],
        )
    ).reshape(-1)
    if forward_pixel_size.size == 1:
        forward_pixel_size = np.repeat(forward_pixel_size, 2)

    reflection = np.asarray(
        state_get_with_candidates(
            state,
            [
                f"{ipref}reflection_coefficient/_data",
                f"{ipref}inner_interactions/reflection_coefficient/_data",
            ],
        )
    )
    if reflection.ndim >= 3:
        reflection = state_pick_index(reflection, index=index, min_ndim=3)
    if reflection.ndim < 2:
        raise ValueError(f"Reflection coefficient under '{ipref}reflection_coefficient/_data' must be at least 2D.")

    forward_shape_arr = state_get_optional(
        state,
        [
            f"{ipref}forward_sampling/shape",
            f"{ipref}forward_sampling/_shape",
        ],
    )
    if forward_shape_arr is not None:
        forward_shape_arr = np.asarray(forward_shape_arr).reshape(-1)
        if forward_shape_arr.size == 1:
            forward_shape_arr = np.repeat(forward_shape_arr, 2)
        if forward_shape_arr.size < 2:
            raise ValueError(
                f"Forward sampling shape under '{ipref}forward_sampling/shape' must provide at least one value."
            )
        forward_shape = tuple(int(v) for v in forward_shape_arr[:2])
    else:
        forward_shape = tuple(int(v) for v in probe.sampling.shape[:2])

    forward_sampling = SamplingGrid.from_tuples(forward_shape, tuple(float(v) for v in forward_pixel_size[:2]))

    output_coordinate_system = CoordinateSystem(
        rotation=sample_rotation,
        translation=probe.coordinate_system.translation,
    )

    return CoherentField(
        data=jnp.zeros((*forward_shape, 1), dtype=probe.data.dtype),
        wavelength=probe.wavelength,
        sampling=forward_sampling,
        coordinate_system=output_coordinate_system,
        propagation_direction=reflected_direction,
    )
