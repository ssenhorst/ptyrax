import functools
import logging
import os
import pathlib
from typing import Any, Callable, Iterable, Tuple

import equinox as eqx
import gin
import jax
import jax.numpy as jnp
import optax
from jax.tree import map as tree_map
from jaxtyping import Array, Float, Integer, Key
from tensorboardX import SummaryWriter
from tqdm.auto import trange

from ptyrax.dataset import ImageDataset, from_hdf5
from ptyrax.hdf5_checkpoint import save_model_hdf5
from ptyrax.logger import log_on_batch_end, log_on_epoch_end, log_on_train_start, tensorboard_to_hdf5
from ptyrax.models.ptychography import (
    ImagePredictionModel,
    NamedLoss,
    PtychographyModel,
    load_model_from_reconstruction,
    preprocess_model,
)
from ptyrax.training import (
    initialize_optimizer_and_state,
    loss,
)


@functools.partial(jax.jit, static_argnums=(0, 1))
def make_batches(n: int, batch_size: int, shuffle_key: Key) -> Integer[Array, " num_batches batch_size"]:
    """Create shuffled batch index arrays for one epoch.

    Args:
        n: Total number of samples (must be divisible by ``batch_size``).
        batch_size: Number of samples per batch.
        shuffle_key: JAX PRNG key for random permutation.

    Returns:
        Integer array of shape ``(n // batch_size, batch_size)`` containing
        shuffled sample indices.
    """
    idx = jnp.arange(n)
    idx = jax.random.permutation(shuffle_key, idx)
    batches = idx.reshape(-1, batch_size)
    return batches


def train_epoch(
    model: ImagePredictionModel,
    dataset: ImageDataset,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    step: int,
    epoch: int,
    epoch_logger: Callable,
    *,
    key: Key,
) -> Tuple[ImagePredictionModel, Any, int, int]:
    """Run a single training epoch over all batches.

    Shuffles the dataset into batches and performs one gradient update per
    batch using :func:`jax.lax.scan`.

    Args:
        model: Current model state.
        dataset: Training dataset containing measured images.
        optimizer: Optax gradient transformation.
        optimizer_state: Current optimizer state.
        step: Global training step counter (unused, reserved).
        epoch: Current epoch index (0-based).
        epoch_logger: Callback invoked at end of epoch with
            ``(epoch, model, total_loss, losses, ordered=False)``.
        key: JAX PRNG key for batch shuffling.

    Returns:
        Tuple of ``(model, optimizer_state, step, epoch)`` with updated values.
    """
    measured_images = dataset.images
    try:
        batch_size = gin.get_bindings("ptyrax.batch.batch_size")
    except ValueError:
        batch_size = 1

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def do_scan(
        model: ImagePredictionModel,
        optimizer_state: optax.OptState,
        measured_images: Float[Array, "n h w"],
        batches: Integer[Array, " num_batches batch_size"],
    ) -> Tuple[Tuple[ImagePredictionModel, Any], Tuple[Float[Array, " b"], list[NamedLoss]]]:
        # We include measured_diffraction_patterns as an argument to avoid closing over it.
        # This causes long compilation times.
        def minibatch_step(
            carry: tuple[ImagePredictionModel, optax.OptState], batch_index: Float[Array, " batch_size"]
        ) -> tuple[tuple[ImagePredictionModel, optax.OptState], tuple[Float[Array, " batch_size"], list[NamedLoss]]]:
            model, optimizer_state = carry
            scanning_position_index = batch_index
            measured_image_batch = measured_images[batch_index]
            scanning_position_index = jnp.asarray(scanning_position_index)
            (loss, losses), model, optimizer_state = train_batch(
                model,
                scanning_position_index,
                measured_image_batch,
                optimizer,
                optimizer_state,
            )

            return (model, optimizer_state), (loss, losses)

        return jax.lax.scan(
            minibatch_step,
            (model, optimizer_state),
            batches,
        )

    batches = make_batches(dataset.n, batch_size, key)
    (model, optimizer_state), (loss, losses) = do_scan(model, optimizer_state, measured_images, batches)
    epoch += 1
    epoch_logger(epoch, model, jnp.sum(loss), losses, ordered=False)

    return model, optimizer_state, step, epoch


@gin.configurable()
def train_session(
    model: ImagePredictionModel,
    dataset: ImageDataset,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    batch_logger: Callable,
    epoch_logger: Callable,
    num_epochs: int = 30,
    epoch_callbacks: Iterable[Callable] = (),
    initial_epoch: int = 0,
    *,
    key: Key,
    **kwargs,  # Added to be backwards compatible with old gin configs
) -> Tuple[ImagePredictionModel, optax.OptState]:
    """Trains a model on a ptychogram dataset. The model is trained for a
    number of epochs, with a given learning rate.

    Args:
        model (PtychographyModel): The model to train.
        ptychogram (Ptychogram): The dataset to train on.
        batch_logger (Callable): A function that logs the loss for each batch.
        epoch_logger (Callable): A function that logs the state of the model at the end of each epoch.
        num_epochs (int): The number of epochs to train for.
        epoch_callbacks (Iterable[Callable]): A list of callback functions which are run after each epoch.
        initial_epoch (int): The epoch to start training from.
    """

    step = 0
    epoch_logger(0, model, epoch_total_loss=None, all_epoch_losses=None, ordered=False)
    dataset.to_gpu()
    epoch = 0
    for _ in (pbar := trange(initial_epoch, initial_epoch + num_epochs, desc="Epoch")):
        key = jax.random.fold_in(key, epoch)
        try:
            train_key, key = jax.random.split(key)
            model, optimizer_state, step, epoch = train_epoch(
                model,
                dataset,
                optimizer,
                optimizer_state,
                step,
                epoch,
                epoch_logger,
                key=train_key,
            )
            *callback_keys, key = jax.random.split(key, len(epoch_callbacks) + 1)
            for callback, callback_key in zip(epoch_callbacks, callback_keys):
                callback_output = callback(model, optimizer_state=optimizer_state, epoch=epoch, key=callback_key)
                if isinstance(callback_output, tuple):
                    # TODO find more elegant way to have callbacks output more variables
                    model, optimizer_state = callback_output
                else:
                    model = callback_output
            pbar.set_postfix()
        except KeyboardInterrupt:
            logging.warning("KeyboardInterrupt received. Returning current model state.")
            return model, optimizer_state
    return model, optimizer_state


def train_model(
    model: PtychographyModel,
    dataset: ImageDataset,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    writer: SummaryWriter,
    *,
    key: Key = jax.random.PRNGKey(42),
    **kwargs,
) -> Tuple[PtychographyModel, optax.OptState]:
    """Top-level training loop that logs initial state and delegates to
    :py:func:`~ptyrax.reconstruct.train_session`.

    Args:
        model: Model to train.
        dataset: Training dataset.
        optimizer: Optax gradient transformation.
        optimizer_state: Initial optimizer state.
        writer: TensorBoard SummaryWriter for logging.
        key: JAX PRNG key.
        **kwargs: Extra keyword arguments forwarded to ``train_session``.

    Returns:
        Tuple of ``(trained_model, trained_optimizer_state)``.
    """
    log_on_train_start(writer, dataset)
    batch_logger = functools.partial(log_on_batch_end, writer)
    epoch_logger = functools.partial(log_on_epoch_end, writer)

    model, optimizer_state = train_session(
        model,
        dataset,
        optimizer,
        optimizer_state,
        batch_logger,
        epoch_logger,
        key=key,
        **kwargs,
    )
    return model, optimizer_state


def post_training(
    model: PtychographyModel,
    log_dir: str,
    output_file: str,
    save_equinox_model: bool = True,
) -> None:
    """Save model outputs after training completes.

    Writes the model as HDF5 and optionally as an equinox binary (``.eqx``),
    and converts TensorBoard logs to an HDF5 archive.

    Args:
        model: Trained model to serialize.
        log_dir: Directory containing TensorBoard event files.
        output_file: Destination HDF5 file for model parameters.
        save_equinox_model: Also save raw ``.eqx`` binary.
    """
    save_model_hdf5(model, output_file)
    if save_equinox_model:
        model.save(f"{os.path.splitext(output_file)[0]}.eqx")

    # Save tensorboard logs to log_dir and also to output directory
    tensorboard_to_hdf5(log_dir, os.path.join(log_dir, "tensorboard_logs.hdf5"))

    if output_dir := os.path.dirname(output_file):
        output_tensorboard = os.path.join(output_dir, "tensorboard_logs.hdf5")
        tensorboard_to_hdf5(log_dir, output_tensorboard)


@gin.configurable
def reconstruct(
    dataset_path: str,
    output_file: pathlib.Path,
    log_dir: str,
    dataset_load_fn: Callable[[str], ImageDataset] = from_hdf5,
    preprocess_functions: list[Callable[[ImageDataset], ImageDataset]] = (),
    continue_from_reconstruction: str | None = None,
    sweep_id: str = None,
    *,
    key: Key,
    model_type: type[ImagePredictionModel] = PtychographyModel,
    **kwargs,
) -> ImagePredictionModel:
    """Trains a model on a ptychogram dataset and saves the results."""
    writer = SummaryWriter(logdir=log_dir)
    logging.info(f"Logging Tensorboard to directory: {log_dir}")

    # Log sweep_id to tensorboard if provided
    if sweep_id is not None:
        writer.add_text("experiment/sweep_id", sweep_id)
        logging.info(f"Sweep ID: {sweep_id}")

    dataset = dataset_load_fn(dataset_path)
    for fn in preprocess_functions:
        dataset = fn(dataset)

    model = model_type.from_image_dataset(dataset, tensorboard_writer=writer)
    if continue_from_reconstruction is not None:
        logging.info(f"Continuing from {continue_from_reconstruction}")
        model = load_model_from_reconstruction(dataset, continue_from_reconstruction)

    model = preprocess_model(model)
    logging.info(f"Initialized model: \n {model}")

    optimizer_state, partitioned_optimizer = initialize_optimizer_and_state(model)

    trained_model, trained_optimizer_state = train_model(
        model,
        dataset,
        partitioned_optimizer,
        optimizer_state,
        writer,
        key=key,
        **kwargs,
    )

    post_training(trained_model, log_dir, output_file)
    return trained_model


@eqx.filter_jit
def train_batch(
    model: ImagePredictionModel,
    scanning_position_index: Integer[Array, " b"],
    measured_diffraction_pattern: Float[Array, "b w h"],
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
) -> Tuple[Tuple[float, Any], ImagePredictionModel, optax.OptState]:
    """Perform a single gradient step on one mini-batch.

    Computes the loss and gradient, conjugates the Wirtinger derivative,
    applies the optimizer update, and returns the updated model.

    Args:
        model: Current model state.
        scanning_position_index: Batch of scanning position indices.
        measured_diffraction_pattern: Measured images for this batch.
        optimizer: Optax gradient transformation.
        optimizer_state: Current optimizer state.

    Returns:
        Tuple of ``((total_loss, named_losses), model, optimizer_state)``.
    """
    with jax.debug_nans(True):
        losses, gradient = eqx.filter_value_and_grad(loss, has_aux=True)(
            model, scanning_position_index, measured_diffraction_pattern
        )
    # JAX takes the complex-conjugate version of the Wirtinger derivative, so we must undo this for the update.
    gradient = tree_map(jnp.conjugate, gradient)
    updates, optimizer_state = optimizer.update(gradient, optimizer_state)
    model = eqx.apply_updates(model, updates)
    return losses, model, optimizer_state
