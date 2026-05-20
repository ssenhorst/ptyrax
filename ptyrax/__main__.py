import argparse
import contextlib
import datetime
import functools
import logging
import os
import pathlib
import random
import shutil
import sys
import tempfile
import warnings
from typing import Callable, Literal

import gin
import jax
import numpy as np
from jaxtyping import Key
from omegaconf import OmegaConf

from ptyrax.config import gin_bind_from_scoped_dict, load_yaml_config, resolve_config_paths
from ptyrax.experiment import run_experiment
from ptyrax.reconstruct import reconstruct
from ptyrax.simulate import simulate


@gin.configurable
def main(sweep_id: str = None, *args, **kwargs) -> None:
    """Main entry point for the script. Parses arguments and runs the ptyrax
    function.

    Args:
        sweep_id: Optional sweep ID for experiment tracking (can be set via gin config)

    Returns:
        None
    """
    parsed_args = parse_arguments()
    configure_logging(verbose=parsed_args.verbose)
    if parsed_args.mode == "experiment":
        run_experiment(parsed_args.experiment_name, dry_run=parsed_args.dry_run)
        return
    elif parsed_args.mode in ("simulate", "reconstruct"):
        parsed_args.simulate = parsed_args.mode == "simulate"
        resolved_configs, temp_config_dir = resolve_config_paths(parsed_args.config)
        try:
            initialize_gin_configs(resolved_configs)
            log_dir, output_file, output_dir, input_file = prepare_filesystem(parsed_args)
            configure_logging(log_dir=log_dir, verbose=parsed_args.verbose)
            for config_path in resolved_configs:
                copy_file_to_directory(config_path, log_dir)
            if parsed_args.simulate:
                for config_path in resolved_configs:
                    copy_file_to_directory(config_path, output_dir)
            save_gin_config_to_directory(log_dir)
            configure_jax_settings()
            key = set_seed()

            # Get sweep_id from gin config parameter, command-line, or function argument
            final_sweep_id = getattr(parsed_args, "sweep_id", None) or sweep_id
            if final_sweep_id is None:
                try:
                    final_sweep_id = gin.query_parameter("main.sweep_id")
                except (ValueError, KeyError):
                    final_sweep_id = None

            def _maybe_profile(func: Callable):  # noqa: F821
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    if parsed_args.profile:
                        logging.info(f"Starting profiler, writing to {log_dir}")
                        with jax.profiler.trace(log_dir):
                            return func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)

                return wrapper

            _ptyrax = _maybe_profile(ptyrax)
            _ptyrax(
                dataset_path=input_file,
                output_file=output_file,
                log_dir=log_dir,
                mode=parsed_args.mode,
                debug=parsed_args.debug,
                sweep_id=final_sweep_id,
                key=key,
            )
        finally:
            if temp_config_dir is not None:
                shutil.rmtree(temp_config_dir, ignore_errors=True)
    else:
        raise ValueError(f"Unknown mode: {parsed_args.mode}. Expected 'reconstruct', 'simulate', or 'experiment'.")


@gin.configurable
def ptyrax(
    dataset_path: str,
    output_file: str,
    log_dir: str,
    mode: Literal["reconstruct", "simulate"] = "reconstruct",
    debug: bool = False,
    sweep_id: str = None,
    *,
    key: Key = jax.random.PRNGKey(0),
) -> None:
    """Main function to run the ptychographic reconstruction.

    Args:
        dataset_path (str): Path to the source dataset file.
        output_file (str): Path to the output file.
        log_dir (str): Directory for logging.
        ptychogram_load_fn (callable): Function to load the ptychogram. Defaults to Ptychogram.from_file.
        preprocess_functions (tuple): Tuple of preprocessing functions to apply to the ptychogram.
                                    Defaults to an empty tuple.
        seed (int): Random seed to use for batch shuffling. If None, no seed is used. Defaults to None.
        sweep_id (str): Optional sweep ID for experiment tracking.

    Returns:
        None
    """
    if debug:
        enable_debug_mode()
    output_file = pathlib.Path(output_file)
    if mode == "simulate":
        simulate(output_file, key=key, dataset_path=dataset_path, sweep_id=sweep_id, log_dir=log_dir)
    elif mode == "reconstruct":
        reconstruct(dataset_path, output_file, log_dir, key=key, sweep_id=sweep_id)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'reconstruct' or 'simulate'.")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments with subcommands for different modes.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="Ptyrax",
        description="Reconstructs ptychographic datasets using a JAX-based autograd backend.",
        epilog="Developed by: Sander Senhorst",
    )
    global_parser = argparse.ArgumentParser(add_help=False)

    # Shared options
    global_parser.add_argument("-c", "--config", help="Path or HTTP(S) URL to the configuration file.", action="append")
    global_parser.add_argument("-l", "--log_dir", help="Directory for logging.")
    global_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    global_parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")

    # Subparsers for modes
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Operation mode")

    # Reconstruct mode
    reconstruct_parser = subparsers.add_parser("reconstruct", help="Run reconstruction", parents=[global_parser])
    reconstruct_parser.add_argument("dataset_file", help="Path to the ptychogram file.")
    reconstruct_parser.add_argument("output_file", nargs="?", default=None, help="Path to the output file.")
    reconstruct_parser.add_argument("--profile", action="store_true", help="Enable profiling mode.")
    reconstruct_parser.add_argument("--sweep_id", default=None, help="Sweep ID for experiment tracking.")

    # Simulate mode
    simulate_parser = subparsers.add_parser("simulate", help="Run simulation", parents=[global_parser])
    simulate_parser.add_argument("output_file", nargs="?", default=None, help="Path to the output file.")
    simulate_parser.add_argument("--dataset_file", help="Optional path to use as base dataset file.", default=None)
    simulate_parser.add_argument("--profile", action="store_true", help="Enable profiling mode.")
    simulate_parser.add_argument("--sweep_id", default=None, help="Sweep ID for experiment tracking.")

    # Experiment mode
    experiment_parser = subparsers.add_parser(
        "experiment", help="Run experiments / queue DVC sweeps", parents=[global_parser]
    )
    experiment_parser.add_argument("experiment_name", help="The name of the dvc experiment to run.")
    experiment_parser.add_argument("--dry-run", action="store_true", help="Print experiments without queuing")

    args = parser.parse_args()
    return args


def enable_debug_mode() -> None:
    """Enable JAX debug settings and strict Python warnings.

    Sets ``JAX_DEBUG_NANS``, ``JAX_DISABLE_JIT``, and ``JAX_DEBUG_INFS``
    environment variables, and raises warnings as errors.
    """
    os.environ["JAX_DEBUG_NANS"] = "true"
    os.environ["JAX_DISABLE_JIT"] = "true"
    os.environ["JAX_DEBUG_INFS"] = "true"
    logging.getLogger().setLevel(logging.DEBUG)
    warnings.simplefilter("error")
    logging.debug("Debug mode enabled: JAX_DEBUG_NANS and JAX_DISABLE_JIT set.")


def copy_file_to_directory(file_path: str, output_dir: str) -> None:
    """Copy a file to a directory, preserving metadata.

    Args:
        file_path: Source file path.
        output_dir: Destination directory.
    """
    file_name = os.path.basename(file_path)
    destination = os.path.join(output_dir, file_name)
    shutil.copy2(file_path, destination)


def save_gin_config_to_directory(directory: str) -> None:
    """Write the current gin configuration string to a file in ``directory``.

    Args:
        directory: Target directory where ``full_gin_config.gin`` is written.
    """
    gin_str = gin.config_str()
    with open(os.path.join(directory, "full_gin_config.gin"), "w") as f:
        f.write(gin_str)


@gin.configurable()
def set_seed(seed: int = 42) -> Key:
    """Set global random seeds and return a JAX PRNG key.

    Sets seeds for NumPy and Python's ``random`` module, then returns
    a :func:`jax.random.PRNGKey`.

    Args:
        seed: Integer seed value.

    Returns:
        JAX PRNG key initialized from ``seed``.
    """
    # Magic number 42 chosen by Douglas Adams
    logging.info(f"Setting random seed to {seed}")
    np.random.seed(seed)
    random.seed(seed)
    key = jax.random.PRNGKey(seed)
    return key


def prepare_filesystem(args: argparse.Namespace) -> tuple[str, str, str, str | None]:
    """Create log and output directories based on parsed CLI arguments.

    Derives log directory names from timestamps and optional tags,
    creates the directories, and returns resolved paths.

    Args:
        args: Parsed CLI namespace with ``dataset_file``, ``log_dir``,
            and ``output_file`` attributes.

    Returns:
        Tuple of ``(log_dir, output_file, output_dir, input_file)``.
    """
    input_path = args.dataset_file
    if args.dataset_file is None:
        try:
            input_path = gin.get_bindings("ptyrax")["dataset_file"]
        except KeyError:
            input_path = None

    filename = os.path.splitext(os.path.basename(input_path))[0] if input_path is not None else "ptyrax"
    args.log_dir = args.log_dir if args.log_dir is not None else f"logs/{filename}/"
    now = datetime.datetime.now()
    try:
        tag = gin.get_bindings("main")["tag"]
    except KeyError:
        tag = None
    tag = f"{os.environ['DVC_EXP_NAME']}_{tag}" if "DVC_EXP_NAME" in os.environ and tag is not None else tag
    experiment_name = f"{now:%Y_%m_%d_%H_%M_%S}_{tag}" if tag is not None else f"{now:%Y_%m_%d_%H_%M_%S}"
    log_dir = os.path.join(args.log_dir, f"{experiment_name}/")
    output_file = args.output_file if args.output_file is not None else f"{filename}_reconstruction.hdf5"
    os.makedirs(log_dir, exist_ok=True)
    output_dir = os.path.dirname(output_file)
    with contextlib.suppress(FileNotFoundError):
        os.makedirs(output_dir, exist_ok=True)
    return log_dir, output_file, output_dir, input_path


def initialize_gin_configs(config_files: list[str]) -> None:
    """Parse and bind gin configuration from YAML or legacy .gin files.

    Supports merging multiple YAML configs via OmegaConf. Legacy ``.gin``
    format triggers a deprecation warning.

    Args:
        config_files: List of config file paths (YAML or .gin).

    Raises:
        ValueError: If no valid config files are provided.
    """
    import ptyrax.external_configurables  # noqa: F401
    import ptyrax.models.ptychography  # noqa: F401

    configs = []
    if config_files is None or not config_files:
        raise ValueError("At least one config file must be specified.")
    for config_file in config_files:
        if config_file.endswith(".gin"):
            warnings.warn(
                "Using .gin config files is deprecated and will be removed in future versions. "
                "Please switch to .yaml or .yml config files."
            )
            gin.parse_config_file(config_file)
            if len(config_files) > 1:
                warnings.warn(
                    "Multiple config files specified, but at least one uses the old .gin format. "
                    "When using .gin files, only a single config file is supported. "
                    f"Parsing only {config_file} and ignoring the rest."
                )
            return
        elif config_file.endswith(".yaml") or config_file.endswith(".yml"):
            yaml_cfg = load_yaml_config(config_file)
            configs.append(yaml_cfg)
    if not configs:
        raise ValueError(
            f"No valid configuration files found. Please provide at least one .yaml or .yml config file. Got "
            f"{config_files}"
        )
    total_config = OmegaConf.merge(*configs)
    try:
        gin_bind_from_scoped_dict(total_config)
    except ValueError as e:
        logging.error(f"Error binding gin configurations: {e}")
        e.add_note(
            "Please check that all configuration keys match the expected gin parameters. "
            "If you just updated ptyrax, some configuration keys may have changed."
        )  # noqa: E501
        raise e


def configure_logging(log_dir: str = None, verbose: bool = False) -> None:
    """Set up Python logging to stdout and optionally to a log file.

    Args:
        log_dir: If provided, a file handler is added writing to
            ``<log_dir>/log.log``.
        verbose: If True, set log level to DEBUG; otherwise INFO.
    """
    logger = logging.getLogger()
    logger.handlers.clear()
    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    configure_logging_handler(stdout_handler, log_level, formatter, logger)
    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, "log.log"))
        configure_logging_handler(file_handler, log_level, formatter, logger)
    logging.info("Logging to console" + (f" and {log_dir}" if log_dir is not None else ""))
    logging.info(f"log level: {logging._levelToName[log_level]} " + (f" and {log_dir}" if log_dir is not None else ""))


def configure_logging_handler(
    handler: logging.Handler, log_level: int, formatter: logging.Formatter, logger: logging.Logger
) -> None:
    """Attach a handler with given level and formatter to a logger.

    Args:
        handler: Logging handler to configure.
        log_level: Logging level to set.
        formatter: Formatter to apply.
        logger: Logger to attach the handler to.
    """
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def configure_jax_settings() -> None:
    """Configure JAX compilation caching for the session.

    Creates a temporary directory for the compilation cache and sets JAX
    config options for persistent caching.
    """
    tmp_dir = tempfile.mkdtemp()
    jax.config.update("jax_compilation_cache_dir", os.path.join(tmp_dir, "jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


if __name__ == "__main__":
    main()
