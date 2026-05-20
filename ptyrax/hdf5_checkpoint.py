from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Literal, NamedTuple, Optional, Tuple, Type
from urllib.parse import quote

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, PyTree

from ptyrax.utils import (
    compile_policy_patterns,
    join_hdf5_paths,
    normalize_hdf5_path,
    resize_to_match,
    warn_if_duplicate_normalized_keys,
)


def _encode_component(s: str) -> str:
    return quote(s, safe="-_.~:+=@")


def _keypath_to_components(path: Iterable[jax.tree_util.KeyEntry]) -> List[str]:
    comps: List[str] = []
    for key in path:
        get_attr_key = jax.tree_util.GetAttrKey
        dict_key = jax.tree_util.DictKey
        sequence_key = jax.tree_util.SequenceKey

        if isinstance(key, get_attr_key):
            comps.append(_encode_component(str(key.name)))
        elif isinstance(key, dict_key):
            comps.append(_encode_component(str(key.key)))
        elif isinstance(key, sequence_key):
            comps.append(_encode_component(str(key.idx)))
        else:
            comps.append(_encode_component(str(key)))
    return comps


def _key_for_path(path: Iterable[jax.tree_util.KeyEntry]) -> str:
    comps = _keypath_to_components(path)
    return "/".join(comps) if comps else ""


def _is_param_leaf(x: ArrayLike) -> bool:
    return eqx.is_array_like(x) and not getattr(x, "_is_static", False)


def _logical_dtype_and_payload(arr: np.ndarray) -> Tuple[str, np.ndarray]:
    dt = arr.dtype
    is_bf16 = (str(dt) == "bfloat16") or (dt == getattr(jnp, "bfloat16", object))
    if is_bf16:
        return "bfloat16", arr.astype(np.float32, copy=False)
    return str(dt), arr


def _restore_logical_dtype(payload: np.ndarray, logical_dtype: str, target_dtype: Optional[np.dtype]) -> np.ndarray:
    out = payload
    if logical_dtype == "bfloat16":
        out = out.astype(np.float32, copy=False)
        out = out.astype(jnp.bfloat16) if target_dtype == jnp.bfloat16 else out.astype(np.dtype("bfloat16"))
    if target_dtype is not None:
        try:
            out = out.astype(target_dtype, copy=False)
        except (TypeError, ValueError) as exc:
            logging.warning(
                "Could not cast array %s to target dtype %s. Continuing unadjusted. Full error:\n%s",
                out,
                target_dtype,
                exc,
            )
    return out


def save_model_hdf5(
    model: PyTree,
    file_path: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
    shuffle: bool = True,
    fletcher32: bool = False,
) -> None:
    """Serialize an equinox model's array leaves to an HDF5 file.

    Each array-like leaf in the model PyTree is stored as a dataset under a
    ``params`` group, using the JAX key-path as the HDF5 hierarchy.

    Args:
        model: Equinox model (or any JAX PyTree) to serialize.
        file_path: Destination HDF5 file path. Parent directories are created
            automatically.
        metadata: Optional metadata dict stored as JSON in the file attributes.
        compression: HDF5 compression filter name (e.g. ``"gzip"``).
        compression_opts: Compression level (0–9 for gzip).
        shuffle: Enable HDF5 shuffle filter for better compression.
        fletcher32: Enable Fletcher32 checksum on datasets.
    """
    pairs, _ = jax.tree_util.tree_flatten_with_path(model)

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with h5py.File(file_path, "a") as f:
        f.attrs["format"] = "equinox_hdf5_state_v1"
        f.attrs["created"] = int(time.time())
        if metadata is not None:
            f.attrs["meta_json"] = json.dumps(metadata, ensure_ascii=False)

        if "params" in f:
            del f["params"]
        g_params = f.create_group("params")

        for keypath, leaf in pairs:
            if not _is_param_leaf(leaf):
                continue
            arr = np.asarray(leaf)
            logical_dtype, payload = _logical_dtype_and_payload(arr)

            rel_path = _key_for_path(keypath) or "_root"
            comps = rel_path.split("/")
            grp = g_params
            for comp in comps[:-1]:
                grp = grp.require_group(comp)
            name = comps[-1]

            is_scalar = payload.shape == ()
            if is_scalar:
                dset = grp.create_dataset(name, data=payload)
            else:
                dset = grp.create_dataset(
                    name,
                    data=payload,
                    compression=compression,
                    compression_opts=compression_opts,
                    shuffle=shuffle,
                    fletcher32=fletcher32,
                )

            dset.attrs["shape"] = list(arr.shape)
            dset.attrs["logical_dtype"] = logical_dtype


def _collect_datasets_recursive(g: h5py.Group, prefix: str = "") -> Dict[str, h5py.Dataset]:
    out: Dict[str, h5py.Dataset] = {}
    for name, item in g.items():
        current = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Dataset):
            out[current] = item
        elif isinstance(item, h5py.Group):
            out |= _collect_datasets_recursive(item, current)
    return out


def load_hdf5_state(
    file_path: str | os.PathLike | h5py.File | h5py.Group,
    *,
    params_root: str = "params",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load flat parameter state from an HDF5 file.

    Args:
        file_path: Path to the HDF5 file, or an already-open
            :class:`h5py.File` / :class:`h5py.Group`.
        params_root: Name of the HDF5 group containing parameters.

    Returns:
        Tuple of ``(state, metadata)`` where *state* maps relative HDF5 paths
        to NumPy arrays, and *metadata* is a dict parsed from the file’s
        ``meta_json`` attribute.
    """
    if isinstance(file_path, (h5py.File, h5py.Group)):
        h5_ctx = None
        root = file_path
    else:
        h5_ctx = h5py.File(file_path, "r")
        root = h5_ctx

    try:
        return _extract_state_from_root(root, params_root)
    finally:
        if h5_ctx is not None:
            h5_ctx.close()


def _extract_state_from_root(root: h5py.Group, params_root: str):
    meta = {}
    if "meta_json" in root.attrs:
        try:
            meta = json.loads(root.attrs["meta_json"])
        except (json.JSONDecodeError, TypeError, ValueError):
            meta = {"_raw_meta_json": str(root.attrs["meta_json"])}

    if params_root_clean := params_root.strip("/"):
        if params_root_clean not in root:
            logging.warning("params_root '%s' was not found in HDF5 input; returning empty state.", params_root)
            return {}, meta
        params_group = root[params_root_clean]
        if not isinstance(params_group, h5py.Group):
            raise ValueError(f"Expected group at params_root='{params_root}', got dataset")

    elif isinstance(root, h5py.Group):
        params_group = root
    else:
        raise ValueError("params_root='' requires `file_path` to be an h5py.Group")
    state: Dict[str, np.ndarray] = {}
    ds_map = _collect_datasets_recursive(params_group)
    for rel_path, dset in ds_map.items():
        payload = dset[()]
        logical_dtype = dset.attrs.get("logical_dtype", str(payload.dtype))
        arr = _restore_logical_dtype(payload, logical_dtype, target_dtype=None)
        state[rel_path] = np.asarray(arr)
    return state, meta


def save_hdf5_state(file_path: str, state: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    """Write a flat parameter state dictionary to an HDF5 file.

    Args:
        file_path: Destination HDF5 file path.
        state: Mapping of relative HDF5 paths to NumPy arrays.
        meta: Metadata dictionary serialized as JSON in file attributes.
    """
    with h5py.File(file_path, "a") as f:
        try:
            f.attrs["meta_json"] = json.dumps(meta)
        except (TypeError, ValueError):
            f.attrs["meta_json"] = json.dumps({"_raw_meta_repr": str(meta)})

        if "params" in f:
            del f["params"]
        params_group = f.create_group("params")
        for rel_path, array in state.items():
            parts = rel_path.split("/")
            grp = params_group
            for part in parts[:-1]:
                grp = grp.require_group(part)

            dset_name = parts[-1]
            dset = grp.create_dataset(dset_name, data=array)
            dset.attrs["logical_dtype"] = str(array.dtype)


class LoadReport(NamedTuple):
    """Report produced by
    :py:func:`~ptyrax.hdf5_checkpoint.apply_hdf5_to_model`.

    Summarizes which parameters were loaded, which had shape mismatches,
    which were missing from the checkpoint, and which checkpoint entries
    were not consumed.

    Attributes:
        loaded: Paths successfully loaded with (checkpoint_shape, model_shape).
        shape_mismatch: Paths where shapes differed and resizing was applied.
        missing_in_ckpt: Model paths not found in checkpoint, with their shapes.
        extraneous_in_ckpt: Checkpoint paths not matched to any model leaf.
    """

    loaded: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]
    shape_mismatch: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]
    missing_in_ckpt: Dict[str, Tuple[int, ...]]
    extraneous_in_ckpt: Dict[str, Tuple[int, ...]]


def _remove_path_underscores(s: str) -> str:
    return "/".join(part.lstrip("_") for part in s.split("/"))


def apply_hdf5_to_model(
    model: PyTree,
    file_path: str | os.PathLike | h5py.File | h5py.Group,
    *,
    strict: bool = False,
    default_expand_policy: Literal["pad", "repeat", "tile"] = "pad",
    cast: bool = True,
    policy_map: Optional[Dict[str, Dict[str, Any]]] = None,
    remove_path_underscores: bool = True,
    path_prefix: Optional[str] = None,
    params_root: str = "params",
) -> Tuple[Any, LoadReport, Dict[str, Any]]:
    """Load HDF5 checkpoint arrays into an existing model PyTree.

    Matches checkpoint paths to model leaves by key-path, handling shape
    mismatches with configurable resize policies.

    Args:
        model: Target model PyTree whose leaves will be replaced.
        file_path: Path to HDF5 checkpoint file or open group.
        strict: If True, raise on any mismatch/missing entries.
        default_expand_policy: Default resize strategy for expanding axes
            (``"pad"``, ``"repeat"``, or ``"tile"``).
        cast: Whether to cast checkpoint arrays to model leaf dtypes.
        policy_map: Per-path regex → policy dict for fine-grained resize control.
        remove_path_underscores: Strip leading underscores from path components
            before matching.
        path_prefix: Only match checkpoint keys under this prefix.
        params_root: HDF5 group name containing parameters.

    Returns:
        Tuple of ``(new_model, report, metadata)`` where *new_model* has
        checkpoint values applied, *report* is a
        :py:class:`~ptyrax.hdf5_checkpoint.LoadReport`, and *metadata* is
        the checkpoint’s stored metadata dict.

    Raises:
        ValueError: If ``strict=True`` and any mismatches exist.
    """
    state, metadata = load_hdf5_state(file_path, params_root=params_root)
    (
        key_mapping,
        normalized_path_prefix,
        compiled_policy_map,
    ) = _prepare_hdf5_load_context(
        state,
        policy_map=policy_map,
        remove_path_underscores=remove_path_underscores,
        path_prefix=path_prefix,
    )

    pairs, treedef = jax.tree.flatten_with_path(model)
    new_leaves: list[Any] = []

    loaded: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    shape_mismatch: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    missing_in_ckpt: Dict[str, Tuple[int, ...]] = {}
    used_ckpt_keys: set[str] = set()

    for keypath, leaf in pairs:
        if not _is_param_leaf(leaf):
            new_leaves.append(leaf)
            continue

        (
            new_leaf,
            used_ckpt_key,
            load_info,
        ) = _load_leaf_from_state(
            keypath=keypath,
            leaf=leaf,
            state=state,
            key_mapping=key_mapping,
            normalized_path_prefix=normalized_path_prefix,
            compiled_policy_map=compiled_policy_map,
            default_expand_policy=default_expand_policy,
            cast=cast,
            remove_path_underscores=remove_path_underscores,
        )

        new_leaves.append(new_leaf)

        if used_ckpt_key is not None:
            used_ckpt_keys.add(used_ckpt_key)

        if load_info is None:
            continue

        rel_path = load_info["rel_path"]
        if load_info["status"] == "missing":
            missing_in_ckpt[rel_path] = load_info["leaf_shape"]
        elif load_info["status"] == "loaded":
            loaded[rel_path] = (load_info["ckpt_shape"], load_info["leaf_shape"])
        elif load_info["status"] == "shape_mismatch":
            shape_mismatch[rel_path] = (load_info["ckpt_shape"], load_info["leaf_shape"])

    new_model = jax.tree_util.tree_unflatten(treedef, new_leaves)
    report = _build_load_report(state, loaded, shape_mismatch, missing_in_ckpt, used_ckpt_keys)

    _raise_if_strict(strict, report)

    return new_model, report, metadata


def _prepare_hdf5_load_context(
    state: Dict[str, np.ndarray],
    *,
    policy_map: Optional[Dict[str, Dict[str, Any]]],
    remove_path_underscores: bool,
    path_prefix: Optional[str],
) -> Tuple[Dict[str, str], str, list[tuple[Any, Dict[str, Any]]]]:
    """Precompute mappings and policies used during HDF5 -> model loading."""
    warn_if_duplicate_normalized_keys(list(state.keys()), normalize_keys=remove_path_underscores)

    normalized_path_prefix = normalize_hdf5_path(path_prefix)
    compiled_policy_map = compile_policy_patterns(policy_map)

    key_mapping = {_remove_path_underscores(k) if remove_path_underscores else k: k for k in state}
    return key_mapping, normalized_path_prefix, compiled_policy_map


def _load_leaf_from_state(
    *,
    keypath: Iterable[jax.tree_util.KeyEntry],
    leaf: Any,  # noqa: ANN401
    state: Dict[str, np.ndarray],
    key_mapping: Dict[str, str],
    normalized_path_prefix: str,
    compiled_policy_map: list[tuple[Any, Dict[str, Any]]],
    default_expand_policy: Literal["pad", "repeat", "tile"],
    cast: bool,
    remove_path_underscores: bool,
) -> Tuple[Any, Optional[str], Optional[Dict[str, Any]]]:
    """Load a single leaf from the HDF5 state, returning the new leaf and load
    metadata."""
    rel_path = _key_for_path(keypath) or "_root"
    rel_path_cmp = _remove_path_underscores(rel_path) if remove_path_underscores else rel_path
    rel_path_cmp = join_hdf5_paths(normalized_path_prefix, rel_path_cmp)

    leaf_shape = tuple(np.shape(leaf))

    if rel_path_cmp not in key_mapping:
        return (
            leaf,
            None,
            {
                "status": "missing",
                "rel_path": rel_path,
                "leaf_shape": leaf_shape,
            },
        )

    ckpt_key = key_mapping[rel_path_cmp]
    ckpt_arr = state[ckpt_key]

    target_dtype = getattr(leaf, "dtype", None) if cast else None
    arr = ckpt_arr.astype(target_dtype, copy=False) if (cast and target_dtype is not None) else ckpt_arr

    ckpt_shape = tuple(arr.shape)

    if leaf_shape == ckpt_shape:
        return (
            jnp.asarray(arr),
            ckpt_key,
            {
                "status": "loaded",
                "rel_path": rel_path,
                "leaf_shape": leaf_shape,
                "ckpt_shape": ckpt_shape,
            },
        )

    path_policy = next(
        (policy for pattern, policy in compiled_policy_map if pattern.fullmatch(rel_path)),
        None,
    )
    default_policy = path_policy.get("default", default_expand_policy) if path_policy else default_expand_policy
    axis_policies = path_policy.get("axes", {}) if path_policy else {}

    resized = resize_to_match(arr, leaf_shape, axis_policies, default_policy)
    if cast and getattr(leaf, "dtype", None) is not None:
        resized = resized.astype(leaf.dtype, copy=False)

    return (
        jnp.asarray(resized),
        ckpt_key,
        {
            "status": "shape_mismatch",
            "rel_path": rel_path,
            "leaf_shape": leaf_shape,
            "ckpt_shape": ckpt_shape,
        },
    )


def _build_load_report(
    state: Dict[str, np.ndarray],
    loaded: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]],
    shape_mismatch: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]],
    missing_in_ckpt: Dict[str, Tuple[int, ...]],
    used_ckpt_keys: set[str],
) -> LoadReport:
    """Construct the LoadReport including extraneous checkpoint entries."""
    extraneous_in_ckpt = {k: tuple(v.shape) for k, v in state.items() if k not in used_ckpt_keys}
    return LoadReport(
        loaded=loaded,
        shape_mismatch=shape_mismatch,
        missing_in_ckpt=missing_in_ckpt,
        extraneous_in_ckpt=extraneous_in_ckpt,
    )


def _raise_if_strict(strict: bool, report: LoadReport) -> None:
    """Raise an error if strict is True and the report contains any
    inconsistencies."""
    if not strict:
        return

    has_issues = report.shape_mismatch or report.missing_in_ckpt or report.extraneous_in_ckpt
    if not has_issues:
        return

    lines = ["Strict load failed:"]
    if report.missing_in_ckpt:
        lines.append(f"- Missing: {len(report.missing_in_ckpt)}")
    if report.shape_mismatch:
        lines.append(f"- Shape mismatches: {len(report.shape_mismatch)}")
    if report.extraneous_in_ckpt:
        lines.append(f"- Extraneous in ckpt: {len(report.extraneous_in_ckpt)}")
    raise ValueError("\n".join(lines))


def create_model_from_hdf5(model_class: Type[eqx.Module], hdf5_path: os.PathLike, *args, **kwargs) -> eqx.Module:
    """Instantiate a model and immediately load weights from an HDF5
    checkpoint.

    Args:
        model_class: Equinox Module class to instantiate.
        hdf5_path: Path to HDF5 checkpoint file.
        *args: Positional arguments forwarded to ``model_class``.
        **kwargs: Keyword arguments forwarded to ``model_class``.

    Returns:
        Model instance with checkpoint weights applied.
    """
    model = model_class(*args, **kwargs)
    loaded_model, _, _ = apply_hdf5_to_model(model, hdf5_path)
    return loaded_model
