import logging
import os
import pathlib
from typing import Callable

import gin
import jax
import jax.numpy as jnp
from jaxtyping import Key
from tensorboardX import SummaryWriter
from tqdm import tqdm

from ptyrax.dataset import ImageDataset, from_hdf5
from ptyrax.hdf5_checkpoint import save_model_hdf5
from ptyrax.logger import log_on_epoch_end, log_on_train_start
from ptyrax.models.ptychography import (
    ImagePredictionModel,
    PtychographyModel,
    load_model_from_reconstruction,
    preprocess_model,
)
from ptyrax.parametrizations import (
    resolve_parametrizations,
)


@gin.configurable
def simulate(
    output_file: pathlib.Path,
    preprocess_functions: list[Callable[[ImageDataset], ImageDataset]] = (),
    dataset_load_fn: Callable[[str], ImageDataset] = from_hdf5,
    continue_from_reconstruction: str | None = None,
    model_type: type[ImagePredictionModel] = PtychographyModel,
    sweep_id: str = None,
    log_dir: str = None,
    *,
    dataset_path: str | None = None,
    key: Key,
) -> ImageDataset:
    """Run a forward simulation of a ptychography model and save outputs.

    Initializes a model (optionally from an existing dataset or reconstruction),
    simulates diffraction patterns for all scanning positions, and writes
    the resulting dataset to ``output_file``.

    Args:
        output_file: Path to write the simulated dataset HDF5 file.
        preprocess_functions: Preprocessing transforms applied to the loaded
            dataset before model initialization.
        dataset_load_fn: Callable that loads an
            :py:class:`~ptyrax.dataset.ImageDataset` from a path.
        continue_from_reconstruction: Optional path to a prior reconstruction
            from which to load model parameters.
        model_type: Model class to instantiate.
        sweep_id: Optional sweep identifier for experiment tracking.
        log_dir: If provided, TensorBoard logs are written here.
        dataset_path: If provided, the model is initialized to mimic the
            geometry of this existing dataset.
        key: JAX PRNG key.

    Returns:
        The simulated :py:class:`~ptyrax.dataset.ImageDataset`.
    """
    # Log sweep_id if provided
    if sweep_id is not None:
        logging.info(f"Sweep ID: {sweep_id}")

    if dataset_path is None:
        try:
            model = model_type()
        except TypeError as e:
            raise TypeError(
                """Failed to initialize the model. This is likely because of missing parameters in the config file. 
                If running a simulation, consider providing a dataset_path to mimic an existing dataset. 
                This may be a dummy dataset just to initialize the model parameters."""
            ) from e
    else:
        # Initialize the model to mimic an existing experimental dataset
        dataset = dataset_load_fn(dataset_path)
        for fn in preprocess_functions:
            dataset = fn(dataset)
        model = model_type.from_image_dataset(dataset)

    if continue_from_reconstruction is not None:
        logging.info(f"Continuing from {continue_from_reconstruction}")
        model = load_model_from_reconstruction(model, continue_from_reconstruction)

    model = preprocess_model(model)
    logging.info(f"Initialized model: \n {model}")

    simulated_dataset = simulate_model(model, key=key)
    if log_dir is not None:
        writer = SummaryWriter(logdir=log_dir)
        log_on_train_start(writer, simulated_dataset)
        log_on_epoch_end(writer, 0, model, None, None)

    post_simulation(model, simulated_dataset, output_file)
    return simulated_dataset


@gin.configurable()
def simulate_model(
    model: ImagePredictionModel,
    *,
    key: Key = jax.random.PRNGKey(0),
) -> ImageDataset:
    """Generate simulated diffraction patterns from a model.

    Iterates over all scanning indices, resolves parametrizations for each,
    and collects the predicted images into an
    :py:class:`~ptyrax.dataset.ImageDataset`.

    Args:
        model: Initialized model with resolved geometry.
        key: JAX PRNG key (unused but kept for API consistency).

    Returns:
        Simulated dataset containing predicted diffraction images.
    """
    indices = jnp.arange(model.n_indices)
    simulated_images = jnp.zeros((len(indices),) + model.image_shape)

    for index in tqdm(indices):
        resolved_model = resolve_parametrizations(model, index)
        # From this point, all ArrayParametrizations are Arrays and
        # all IndexDependentParametrizations have been resolved
        simulated_data = resolved_model()
        simulated_images = simulated_images.at[index].set(simulated_data)

    simulated_dataset = model.to_image_dataset(simulated_images)
    return simulated_dataset


@gin.configurable()
def post_simulation(
    model: PtychographyModel,
    simulated_dataset: ImageDataset,
    output_file: pathlib.Path | None = None,
    save_equinox_model: bool = True,
    postprocess_functions: list[Callable[[ImageDataset], ImageDataset]] = (),
) -> None:
    """Post-process and persist simulation outputs.

    Applies optional post-processing transforms and saves the simulated
    dataset to HDF5. Optionally saves the ground-truth model as both
    ``.eqx`` and HDF5 files alongside the output.

    Args:
        model: Ground-truth model used for simulation.
        simulated_dataset: Simulated dataset to save.
        output_file: Destination path for the simulated dataset HDF5.
        save_equinox_model: Whether to also save model weights.
        postprocess_functions: Transforms applied to the dataset before saving.
    """
    for func in postprocess_functions:
        simulated_dataset = func(simulated_dataset)

    if output_file is not None:
        simulated_dataset.save(output_file)
        if save_equinox_model:
            output_dir = os.path.dirname(output_file)
            model.save(os.path.join(output_dir, "ground_truth_model.eqx"))  #  Raw parameter binary for quick loading
            save_model_hdf5(model, os.path.join(output_dir, "ground_truth_model.hdf5"))
