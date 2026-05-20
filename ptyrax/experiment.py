import copy
import datetime
import itertools
import logging
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
from ruamel.yaml import YAML


# ----------------------
# Main experiment mode
# ----------------------
def run_experiment(experiment_name: str, dry_run: bool = False) -> None:
    """Run or queue a DVC experiment sweep from a config with ``!range`` tags.

    Parses the DVC stage’s config files, expands ``!range`` tags into a
    Cartesian product of parameter combinations, and queues one DVC
    experiment per combination.

    Args:
        experiment_name: Name of the DVC stage to run.
        dry_run: If True, log what would be queued without actually running.

    Raises:
        ValueError: If the experiment name is invalid or multiple dynamic
            config files are found.
    """
    if not _validate_experiment_name(experiment_name):
        raise ValueError(
            f"Experiment name '{experiment_name}' is invalid. "
            "Only alphanumeric characters, underscores and hyphens are allowed (max length 32)."
        )

    sweep_id = _generate_sweep_id()
    logging.info(f"Generated sweep ID: {sweep_id}")

    config_paths = _get_config_files_from_dvc_stage(experiment_name)

    dynamic_cfg_path = None
    dynamic_ranges = None

    for cfg_path in config_paths:
        with open(cfg_path) as f:
            cfg = yaml.load(f)
        if r := _expand_ranges(cfg):
            if dynamic_cfg_path is not None:
                raise ValueError("Only one config file with ranges is allowed.")
            dynamic_cfg_path = cfg_path
            dynamic_ranges = r

    if dynamic_cfg_path is None or dynamic_ranges is None:
        logging.error("No config file with !range found.")
        return

    with open(dynamic_cfg_path) as f:
        original_cfg = yaml.load(f)

    _save_sweep_metadata(sweep_id, experiment_name, dynamic_ranges)

    shutil.move(dynamic_cfg_path, f"{dynamic_cfg_path}.backup")
    try:
        axis_order = list(dynamic_ranges.keys())
        axis_values = _build_axis_values(dynamic_ranges)
        all_combos = _generate_combinations_with_confirmation(axis_order, axis_values)

        _queue_experiment_variants(
            all_combos=all_combos,
            axis_order=axis_order,
            original_cfg=original_cfg,
            dynamic_cfg_path=dynamic_cfg_path,
            experiment_name=experiment_name,
            sweep_id=sweep_id,
            dry_run=dry_run,
        )
    finally:
        backup = f"{dynamic_cfg_path}.backup"
        if Path(backup).exists():
            shutil.move(backup, dynamic_cfg_path)
    logging.info(f"Queued {len(all_combos)} experiments (dry_run={dry_run}).")


def _build_axis_values(dynamic_ranges: Dict[str, Dict[str, List[Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Validate and build axis value structures from dynamic_ranges."""
    axis_values: Dict[str, List[Dict[str, Any]]] = {}
    for axis, entries in dynamic_ranges.items():
        lens = {len(v) for v in entries.values()}
        if len(lens) != 1:
            raise ValueError(
                f"All !range values in axis '{axis}' must have equal length. "
                f"Got lengths: {[f'{k}: {len(v)}' for k, v in entries.items()]}"
            )
        length = lens.pop()
        axis_values[axis] = [{keypath: arr[i] for keypath, arr in entries.items()} for i in range(length)]
    return axis_values


def _generate_combinations_with_confirmation(
    axis_order: List[str],
    axis_values: Dict[str, List[Dict[str, Any]]],
) -> List[tuple[Dict[str, Any], ...]]:
    """Compute Cartesian product across axes and confirm with the user if
    large."""
    all_combos = list(itertools.product(*[axis_values[a] for a in axis_order]))
    logging.info(f"Generated {len(all_combos)} experiment variants.")
    if len(all_combos) > 20:
        logging.info("Large number of experiments generated; are you sure you want to proceed? (Y/n)?")
        ans = input().strip().lower()
        if ans not in ("y", "yes", ""):
            raise KeyboardInterrupt("Experiment queuing aborted by user.")
    return all_combos


def _queue_experiment_variants(
    all_combos: List[tuple[Dict[str, Any], ...]],
    axis_order: List[str],
    original_cfg: Dict[str, Any],
    dynamic_cfg_path: str,
    experiment_name: str,
    sweep_id: str,
    dry_run: bool,
) -> None:
    """Apply parameter combinations, write configs, and queue DVC
    experiments."""
    for combo in all_combos:
        new_cfg = copy.deepcopy(original_cfg)
        variable_metadata_parts = []

        for d, axis in zip(combo, axis_order):
            variable_metadata_parts.append(f"{axis}_{list(d.values())[0]}")
            for keypath, value in d.items():
                path = keypath.split(".")
                _set_nested_value(new_cfg, path, value)

        variable_metadata = "_".join(variable_metadata_parts)
        tag = new_cfg.get("__main__", {}).get("main", {}).get("tag", "")
        _set_nested_value(new_cfg, ["__main__", "main", "tag"], tag + variable_metadata)

        if "__main__" not in new_cfg:
            new_cfg["__main__"] = {}
        if "main" not in new_cfg["__main__"]:
            new_cfg["__main__"]["main"] = {}
        new_cfg["__main__"]["main"]["sweep_id"] = sweep_id

        with open(dynamic_cfg_path, "w") as f:
            yaml.dump(new_cfg, f)

        cmd = ["dvc", "exp", "run", experiment_name, "--queue"]
        if dry_run:
            logging.info(f"[Dry Run] Would queue experiment {experiment_name} with config: {new_cfg}")
            logging.info(f"[Dry Run] Command: {cmd}")
        else:
            _ = subprocess.run(cmd, check=True, text=True, shell=False)  # noqa: S603
        logging.info(f"Queued {len(all_combos)} experiments (dry_run={dry_run}).")


def _validate_experiment_name(experiment_name: str) -> None:
    exp_name_format = r"^[a-zA-Z0-9_\-]{1,32}$"
    return re.match(exp_name_format, experiment_name) is not None


def _generate_sweep_id() -> str:
    """Generate a unique sweep ID with timestamp and random word."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Simple random words for readability (~100k combinations from ~316 adjectives × ~316 nouns)
    # fmt: off
    adjectives = [
        "quick", "lazy", "happy", "sleepy", "brave", "clever", "wild", "calm", "bright", "dark",
        "bold", "cool", "crisp", "deft", "eager", "fair", "firm", "glad", "grand", "keen",
        "kind", "lean", "mild", "neat", "open", "pale", "pure", "rare", "safe", "soft",
        "tall", "tame", "vast", "warm", "wise", "zany", "agile", "alert", "ample", "avid",
        "basic", "brisk", "civic", "clean", "clear", "close", "cozy", "daily", "dense", "dizzy",
        "dorky", "dusty", "early", "empty", "equal", "exact", "extra", "faint", "fancy", "fatal",
        "fiery", "final", "first", "fixed", "fleet", "foggy", "fresh", "frugal", "funny", "fuzzy",
        "giant", "giddy", "given", "great", "green", "gross", "gusty", "handy", "harsh", "hasty",
        "hefty", "humid", "ideal", "inner", "ionic", "ivory", "jazzy", "jolly", "jumpy", "juicy",
        "large", "legal", "level", "light", "lofty", "loose", "loyal", "lucky", "lunar", "lusty",
        "magic", "major", "maple", "meek", "merry", "minor", "misty", "modal", "moist", "moral",
        "muddy", "murky", "muted", "naive", "naval", "noble", "noisy", "novel", "oaken", "olive",
        "outer", "papal", "pasty", "perky", "petty", "plain", "plush", "polar", "prime", "proud",
        "quiet", "rainy", "rapid", "ready", "regal", "rigid", "rocky", "roomy", "rough", "round",
        "royal", "rural", "rusty", "salty", "sandy", "sharp", "sheer", "shiny", "short", "silky",
        "slick", "smart", "smoky", "snowy", "solar", "solid", "sonic", "sorry", "spare", "spicy",
        "stark", "steep", "stiff", "still", "stony", "stout", "sunny", "super", "sweet", "swift",
        "tangy", "tense", "thick", "tight", "timid", "tiny", "tipsy", "total", "tough", "toxic",
        "tricky", "true", "ultra", "uncut", "undue", "upper", "urban", "usual", "utter", "valid",
        "vital", "vivid", "vocal", "wacky", "wavy", "weary", "weird", "white", "whole", "witty",
        "woody", "wordy", "woven", "wrong", "young", "zesty", "zippy", "adept", "alien", "amber",
        "antic", "balmy", "basal", "bland", "blank", "bleak", "bliss", "blunt", "bonny", "bossy",
        "bound", "bowed", "bulky", "burly", "catchy", "cheap", "chief", "chill", "civil", "comfy",
        "coral", "crazy", "curly", "dainty", "dandy", "dewy", "dingy", "dizzy", "downy", "dried",
        "dumpy", "dusky", "eerie", "elfin", "elite", "every", "fetid", "fishy", "flaky", "flint",
        "fluid", "foamy", "forte", "frail", "frank", "freed", "froze", "gaunt", "giddy", "glassy",
        "gooey", "grainy", "grimy", "gruff", "gutsy", "hairy", "hazel", "heady", "heard", "heavy",
        "huffy", "icing", "inane", "inert", "irate", "itchy", "ivory", "jaded", "jumbo", "kinky"
    ]
    nouns = [
        "fox", "dog", "cat", "bird", "fish", "bear", "wolf", "lion", "tiger", "eagle",
        "hawk", "deer", "hare", "dove", "crow", "swan", "goat", "seal", "frog", "moth",
        "crab", "clam", "newt", "mole", "vole", "wren", "lark", "lynx", "pike", "bass",
        "carp", "wasp", "toad", "slug", "snail", "crane", "finch", "mouse", "otter", "panda",
        "raven", "robin", "shark", "snake", "squid", "stork", "trout", "whale", "zebra", "bison",
        "camel", "cobra", "coral", "egret", "falcon", "gecko", "heron", "hyena", "iguana", "koala",
        "llama", "moose", "okapi", "orca", "osprey", "owl", "perch", "quail", "rhino", "sable",
        "stoat", "swift", "tapir", "viper", "yak", "cedar", "birch", "maple", "oak", "pine",
        "aspen", "elm", "ivy", "fern", "moss", "reed", "sage", "thyme", "basil", "mint",
        "cliff", "creek", "delta", "dune", "fjord", "glade", "gorge", "grove", "knoll", "marsh",
        "oasis", "plain", "ridge", "shoal", "slope", "thorn", "trail", "vale", "bloom", "brush",
        "cloud", "comet", "flare", "flame", "frost", "gleam", "grain", "gust", "jewel", "light",
        "mist", "ocean", "orbit", "pearl", "plume", "prism", "pulse", "quartz", "river", "shell",
        "shore", "spark", "spire", "spray", "star", "steam", "stone", "storm", "surge", "tide",
        "torch", "tower", "twist", "vault", "wave", "wind", "anvil", "arrow", "badge", "blade",
        "bolt", "charm", "crest", "crown", "drum", "flint", "forge", "glyph", "grain", "guard",
        "hatch", "horn", "ingot", "lance", "lever", "medal", "notch", "panel", "pivot", "plate",
        "prong", "quill", "rivet", "rune", "shard", "spear", "spoke", "staff", "stake", "strut",
        "tally", "token", "wedge", "wick", "arch", "basin", "beam", "cairn", "chord", "clasp",
        "cog", "dome", "edge", "facet", "flange", "grate", "hasp", "joint", "keel", "latch",
        "ledge", "link", "loom", "mesh", "node", "petal", "plank", "rail", "ramp", "ring",
        "rotor", "scale", "seam", "shaft", "slab", "slot", "span", "spool", "stem", "strut",
        "tread", "truss", "valve", "vane", "weld", "yoke", "agate", "amber", "beryl", "chalk",
        "clay", "cobalt", "crystal", "ember", "flint", "garnet", "glass", "gold", "jade", "jet",
        "lapis", "lead", "mica", "nickel", "onyx", "opal", "ore", "ruby", "rust", "sand",
        "silk", "slate", "steel", "tin", "topaz", "zinc", "atlas", "axiom", "canon", "cipher",
        "creed", "dogma", "draft", "edict", "epoch", "ethos", "flux", "focus", "glyph", "helix",
        "index", "locus", "maxim", "motif", "nexus", "pivot", "proxy", "realm", "scope", "sigma",
        "theta", "torus", "triad", "unity", "vapor", "verge", "vigor", "volta", "zenith", "zonal"
    ]
    # fmt: on
    random_word = f"{random.choice(adjectives)}-{random.choice(nouns)}"  # noqa: S311
    return f"{timestamp}_{random_word}"


def _save_sweep_metadata(sweep_id: str, stage_name: str, dynamic_ranges: Dict[str, Dict[str, List[Any]]]) -> None:
    """Save sweep metadata to dvc_sweeps.yaml registry."""
    registry_path = Path("dvc_sweeps.yaml")

    # Load existing registry or create new
    if registry_path.exists():
        with open(registry_path) as f:
            registry = yaml.load(f) or {}
    else:
        registry = {}

    # Extract sweep axis definitions (tag name + first value)
    sweep_axes = {}
    for axis, entries in dynamic_ranges.items():
        # Get first value from first entry
        first_entry = next(iter(entries.values()))
        sweep_axes[axis] = first_entry[0] if first_entry else None

    # Add this sweep to registry
    registry[sweep_id] = {
        "stage": stage_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "sweep_axes": sweep_axes,
        "parameter_ranges": {axis: list(entries.keys()) for axis, entries in dynamic_ranges.items()},
    }

    # Save registry
    with open(registry_path, "w") as f:
        yaml.dump(registry, f)


# ----------------------
# Custom YAML tag !range
# ----------------------
class RangeTag:
    """Custom YAML tag representing a parameter sweep range.

    Used in experiment config files with the ``!range`` YAML tag to specify
    values that should be swept over in a parameter study.

    Attributes:
        axis: Name of the sweep axis (used for grouping), or None for auto.
        values: List of values to sweep over.
    """

    def __init__(self, axis: str | None, values: List[Any]) -> None:
        self.axis = axis
        self.values = values


yaml = YAML()
yaml.allow_duplicate_keys = True


# Register !range constructor
def range_constructor(loader: Any, node: Any) -> RangeTag:  # noqa: ANN401
    values = loader.construct_sequence(node)
    return (
        RangeTag(axis=values[0], values=values[1:])
        if isinstance(values, list) and len(values) > 1 and isinstance(values[0], str)
        else RangeTag(axis=None, values=values)
    )


yaml.constructor.add_constructor("!range", range_constructor)


def _range_representer(dumper: Any, data: RangeTag):  # noqa: ANN401
    if data.axis is None:
        return dumper.represent_sequence("!range", data.values)
    else:
        return dumper.represent_sequence("!range", [data.axis] + data.values)


yaml.representer.add_representer(RangeTag, _range_representer)


# ----------------------
# Helper functions
# ----------------------
def _set_nested_value(cfg: Dict[str, Any], path: List[str], value: Any) -> None:  # noqa: ANN401
    """Set a value in a nested dictionary by following a key path.

    Args:
        cfg: Nested dictionary to modify in-place.
        path: List of keys forming the path to the target.
        value: Value to assign at the terminal key.
    """
    for p in path[:-1]:
        cfg = cfg[p]
    cfg[path[-1]] = value


# ----------------------
# Range expansion with axis grouping
# ----------------------
def _expand_ranges(obj: Any, prefix: List[str] | None = None) -> Dict[str, Dict[str, List[Any]]]:  # noqa: ANN401
    """
    Returns dict: axis -> { keypath -> list_of_values }

    Rules:
    • Multiple ranges with the same axis are zipped (must have equal length).
    • Different axes form independent groups whose tensor product is taken.
    """
    if prefix is None:
        prefix = []

    grouped: Dict[str, Dict[str, List[Any]]] = {}

    def add(axis: str, key: str, vals: List[Any]):
        grouped.setdefault(axis, {})[key] = vals

    # Dict recursion
    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = _expand_ranges(v, prefix + [k])
            for axis, d in sub.items():
                for kk, vv in d.items():
                    add(axis, kk, vv)
        return grouped

    # List: detect RangeTag within
    if isinstance(obj, list):
        tags = [x for x in obj if isinstance(x, RangeTag)]
        if tags:
            base = [x for x in obj if not isinstance(x, RangeTag)]
            for tag in tags:
                axis = tag.axis or f"__axis_{'.'.join(prefix)}"
                # All zipped:
                expanded = [base + [v] for v in tag.values]
                add(axis, ".".join(prefix), expanded)
            return grouped
        else:
            # Plain list: recurse elements
            for idx, v in enumerate(obj):
                sub = _expand_ranges(v, prefix + [str(idx)])
                for axis, d in sub.items():
                    for kk, vv in d.items():
                        add(axis, kk, vv)
            return grouped

    # Single RangeTag
    if isinstance(obj, RangeTag):
        axis = obj.axis or f"__axis_{'.'.join(prefix)}"
        add(axis, ".".join(prefix), obj.values)

    return grouped


# ----------------------
# Load DVC stage config
# ----------------------
def _get_config_files_from_dvc_stage(stage_name: str) -> List[str]:
    """Extract YAML config file paths referenced by a DVC stage.

    Reads ``dvc.yaml`` and collects paths from the stage’s ``params``
    and ``deps`` sections.

    Args:
        stage_name: Name of the DVC stage.

    Returns:
        List of YAML file paths used by the stage.
    """
    with open("dvc.yaml") as f:
        full_config = yaml.load(f)

    stage = full_config.get("stages", {}).get(stage_name, {})
    params = stage.get("params", [])
    deps = stage.get("deps", [])

    yaml_files: List[str] = []

    # Params entries are dicts: { file.yaml: null or subkeys }
    for p in params:
        if isinstance(p, dict):
            path = list(p.keys())[0]
            if path.endswith((".yaml", ".yml")):
                yaml_files.append(path)

    yaml_files.extend(d for d in deps if isinstance(d, str) and d.endswith((".yaml", ".yml")))
    return yaml_files


# ----------------------
# HDF5 pooled experiment utilities
# ----------------------
def iter_experiments(hdf5_path: str | Path) -> Generator[tuple[str, Any], None, None]:
    """Iterate over experiments in a pooled HDF5 file.

    Args:
        hdf5_path: Path to pooled experiment HDF5 file

    Yields:
        Tuple of (experiment_name, experiment_group)

    Example:
        >>> for exp_name, exp_group in iter_experiments("experiment_sweep_xyz.hdf5"):
        ...     print(f"Processing {exp_name}")
        ...     losses = exp_group["training/scalars/0_loss/0_loss_total/value"][()]
    """
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        # Get experiment names from index
        exp_names = f["index/exp_names"][()]

        # Convert bytes to strings if needed
        if exp_names.dtype.kind == "S":
            exp_names = [name.decode() for name in exp_names]

        for exp_name in exp_names:
            if exp_name in f:
                yield exp_name, f[exp_name]


def get_experiment_index(hdf5_path: str | Path) -> dict:
    """Load the experiment index as a dictionary.

    Args:
        hdf5_path: Path to pooled experiment HDF5 file

    Returns:
        Dict with keys: 'exp_names', 'indices', 'param_{name}' for each parameter

    Example:
        >>> index = get_experiment_index("experiment_sweep_xyz.hdf5")
        >>> print(index["exp_names"])
        >>> print(index["param_angle"])
    """
    import h5py

    index_data = {}

    with h5py.File(hdf5_path, "r") as f:
        index_group = f["index"]

        for key in index_group.keys():
            data = index_group[key][()]

            # Convert bytes to strings if needed
            if data.dtype.kind == "S":
                data = np.array([item.decode() if isinstance(item, bytes) else item for item in data])

            index_data[key] = data

    return index_data


def load_experiment_by_name(hdf5_path: str | Path, exp_name: str) -> dict:
    """Load a specific experiment by name.

    Args:
        hdf5_path: Path to pooled experiment HDF5 file.
        exp_name: Experiment name to load.

    Returns:
        Nested dict with experiment data (scalars, images, etc.).

    Example:
        >>> exp = load_experiment_by_name("experiment_sweep_xyz.hdf5", "leaky-lulu")
        >>> losses = exp["training"]["scalars"]["0_loss"]["0_loss_total"]["value"]
    """
    import h5py

    def _load_group(group: Any):  # noqa: ANN401
        """Recursively load HDF5 group to dict."""
        result = {}
        for key in group.keys():
            if isinstance(group[key], h5py.Group):
                result[key] = _load_group(group[key])
            else:
                result[key] = group[key][()]

        # Also load attributes
        result["_attrs"] = dict(group.attrs)
        return result

    with h5py.File(hdf5_path, "r") as f:
        if exp_name not in f:
            raise KeyError(f"Experiment '{exp_name}' not found in {hdf5_path}")

        return _load_group(f[exp_name])


def get_sweep_axes(hdf5_path: str | Path) -> dict:
    """Get sweep axis information from pooled HDF5.

    Args:
        hdf5_path: Path to pooled experiment HDF5 file

    Returns:
        Dict with sweep metadata (sweep_id, n_experiments, parameter names)

    Example:
        >>> axes = get_sweep_axes("experiment_sweep_xyz.hdf5")
        >>> print(axes["sweep_id"])
        >>> print(axes["param_names"])
    """
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        # Get attributes
        sweep_info = {
            "sweep_id": f.attrs.get("sweep_id", "unknown"),
            "n_experiments": f.attrs.get("n_experiments", 0),
        }

        # Get parameter names from index
        param_names = [key.replace("param_", "") for key in f["index"].keys() if key.startswith("param_")]
        sweep_info["param_names"] = param_names

        return sweep_info
