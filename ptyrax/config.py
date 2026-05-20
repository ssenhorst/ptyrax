import logging
import os
import tempfile
import urllib.request
from typing import Any
from urllib.parse import urlparse

import gin
import requests
from omegaconf import DictConfig, ListConfig, OmegaConf
from ruamel.yaml import YAML

from ptyrax.utils import flatten_dict

yaml = YAML(typ="safe", pure=True)


def gin_ref_constructor(loader: YAML, node: Any) -> str:  # noqa: ANN401
    """Keep !@ refs as tagged strings for OmegaConf compatibility."""
    value = loader.construct_scalar(node)
    return f"!@:{value}"


@gin.configurable()
def range_ref_constructor(loader: YAML, node: Any, sweep_indices: list[str, int] = None) -> str:  # noqa: ANN401
    """Keep only the first value for !range tags."""
    seq = loader.construct_sequence(node)
    seq_values = seq[1:] if type(seq[0]) is str else seq

    if sweep_indices is not None:
        # Sweep indices should ideally be a dict, but this is not compatible with the current YAML constructor API, so
        # we use a list of tuples instead.
        sweep_dict = dict(sweep_indices)
        if seq[0] in sweep_dict:
            index = sweep_dict[seq[0]]
            if index < len(seq_values):
                return seq_values[index]
            else:
                logging.warning(
                    "!range tag had sweep index %d but only %d values. Using last value.",
                    index,
                    len(seq_values),
                )
                return seq_values[-1]

    if len(seq_values) > 1:
        logging.warning(
            f"!range tag {seq[0]} had multiple values in yaml file. Only the first will be used. "
            "The range tag should be used in conjunction with the `ptyrax experiment` command line interface."
        )
    return seq_values[0]  # Note: first entry is the axis


yaml.constructor.add_constructor("!range", range_ref_constructor)
yaml.constructor.add_constructor("!@", gin_ref_constructor)


# --- Recursive parser ---
def parse_maybe_gin_ref(v: Any) -> Any:  # noqa: ANN401
    """Recursively convert tagged strings, lists, and dicts to Gin
    references."""
    if isinstance(v, str) and v.startswith("!@:"):
        ref = v[len("!@:") :]
        return gin.config.parse_value(f"@{ref}")
    elif isinstance(v, list):
        return [parse_maybe_gin_ref(item) for item in v]
    elif isinstance(v, dict):
        return {k: parse_maybe_gin_ref(val) for k, val in v.items()}
    else:
        return v


def gin_bind_from_scoped_dict(cfg: DictConfig | ListConfig, main_scope: str = "__main__") -> None:
    """Convert a Hydra/OmegaConf config into Gin bindings with scopes and
    nested Gin references."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    for scope, params in cfg_dict.items():
        flat_params = flatten_dict(params)
        for key, value in flat_params.items():
            parsed_val = parse_maybe_gin_ref(value)
            full_key = f"{scope}/{key}" if scope != main_scope else key
            gin.bind_parameter(full_key, parsed_val)


def load_yaml_config(yaml_path: str) -> DictConfig | ListConfig:
    """Load YAML into OmegaConf for compatibility."""
    with open(yaml_path, "r") as f:
        data = yaml.load(f)
    return OmegaConf.create(data)


def _is_url(path: str) -> bool:
    return isinstance(path, str) and (path.startswith("http://") or path.startswith("https://"))


def _download_file_from_url(url: str, output_path: str) -> None:
    """Download a remote config file to ``output_path`` with an HTTP(S)
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
        raise RuntimeError(f"Failed to download remote config from '{url}'") from requests_exc


def resolve_config_paths(config_paths: list[str] | None) -> tuple[list[str] | None, str | None]:
    """Resolve local/remote config paths into local file paths.

    Remote HTTP(S) config files are downloaded into a temporary directory.

    Args:
        config_paths: Config paths provided by CLI.

    Returns:
        Tuple of ``(resolved_paths, temp_directory)``.
        ``temp_directory`` is ``None`` when no downloads are needed.
    """
    if config_paths is None:
        return None, None

    if not any(_is_url(path) for path in config_paths):
        return config_paths, None

    temp_dir = tempfile.mkdtemp(prefix="ptyrax_configs_")
    resolved_paths: list[str] = []
    used_names: set[str] = set()

    for idx, config_path in enumerate(config_paths):
        if not _is_url(config_path):
            resolved_paths.append(config_path)
            continue

        parsed = urlparse(config_path, scheme="https")
        base_name = os.path.basename(parsed.path) or f"remote_config_{idx}.yaml"
        stem, ext = os.path.splitext(base_name)
        if ext.lower() not in (".yaml", ".yml", ".gin"):
            ext = ".yaml"
            stem = stem or f"remote_config_{idx}"
        candidate = f"{stem}{ext}"
        suffix = 1
        while candidate in used_names:
            candidate = f"{stem}_{suffix}{ext}"
            suffix += 1
        used_names.add(candidate)

        destination = os.path.join(temp_dir, candidate)
        _download_file_from_url(config_path, destination)
        resolved_paths.append(destination)

    return resolved_paths, temp_dir
