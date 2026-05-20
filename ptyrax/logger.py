from __future__ import annotations

import io
import pathlib
from typing import TYPE_CHECKING, Any, Union

import gin
import h5py
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Inexact

# Decode images into arrays (stack along axis 0)
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter

from ptyrax.parametrizations import resolve_parametrizations
from ptyrax.utils import plot

if TYPE_CHECKING:
    from ptyrax.dataset import ImageDataset
    from ptyrax.models.ptychography import ImagePredictionModel


@gin.configurable
def log_image(writer: SummaryWriter, tag: str, tensor: Inexact[Array, "..."], step: int, **kwargs) -> None:
    """Log an image to tensorboard.

    Args:
        writer: SummaryWriter
            Tensorboard writer
        tag: str
            Tag for the image
        tensor: np.ndarray
            Image to log
        step: int
            Current step
        **kwargs:
            Additional arguments for the plot function

    Returns:
        None
    """
    fig, axs, im = plot(tensor, show=False, **kwargs)
    writer.add_figure(tag, fig, step)


@gin.configurable
def log_on_train_start(writer: SummaryWriter, dataset: ImageDataset, debug: bool = False) -> None:
    """Hook to log initial information at the start of training to TensorBoard.

    Args:
        writer: TensorBoard `SummaryWriter` instance.
        dataset: Dataset to inspect and optionally log example images from.
        debug: If True, log additional debug images for every dataset item.

    Returns:
        None
    """
    writer.add_text("gin config", gin.config_str(max_line_length=70))
    for line in gin.config.config_str().split("\n"):
        if line.startswith("#"):
            continue
        try:
            key, _ = line.split(" = ")
            value = gin.query_parameter(key)
            writer.add_text(key.replace(".", "/"), str(value))
        except ValueError:
            continue
    if debug:
        for i in range(dataset.n):
            pat = dataset[i]
            log_image(writer, "4_model/true_dif_pat", pat, i, gamma=0.5)
    else:
        log_image(writer, "4_model/true_dif_pat/0", dataset[0], 0, gamma=0.5)


@gin.configurable
def log_on_epoch_end(
    writer: SummaryWriter,
    epoch: int,
    model: ImagePredictionModel,
    epoch_total_loss: float,
    all_epoch_losses: dict,
    debug: bool = False,
    diff_pat_gamma: float = 0.5,
    log_every: int = 1,
    **kwargs,
) -> None:
    """Hook executed at the end of each epoch to log predictions, scalars and
    other metrics.

    Args:
        writer: TensorBoard `SummaryWriter`.
        epoch: Current epoch index.
        model: Model used for prediction/logging.
        epoch_total_loss: Total loss for the epoch.
        all_epoch_losses: Iterable of NamedLoss-like objects.
        debug: If True, log per-sample predictions.
        diff_pat_gamma: Gamma applied to diffraction patterns when visualizing.
        log_every: Only run logging every `log_every` epochs.

    Returns:
        None
    """
    if epoch % log_every != 0:
        return
    if hasattr(model, "__log_epoch__"):
        model.__log_epoch__(writer, epoch, **kwargs)

    if debug:
        for i in range(model.n_indices):
            pred_im = resolve_parametrizations(model, jnp.array(i))()
            log_image(writer, f"4_model/pred_dif_pat/{i}", pred_im, epoch, gamma=diff_pat_gamma)
    else:
        pred_im = resolve_parametrizations(model, jnp.array(0))()
        log_image(writer, "4_model/pred_dif_pat/0", pred_im, epoch, gamma=diff_pat_gamma)

    if epoch_total_loss is not None:
        writer.add_scalar("0_loss/0_loss_total", epoch_total_loss, epoch)

    if all_epoch_losses is not None:
        for loss in all_epoch_losses:
            value = np.array(loss.value)
            N = value.shape[0]
            for i, v in enumerate(value):
                writer.add_scalar(f"0_loss/1_loss_terms/{loss.tag}", v, epoch * N + i)

    writer.flush()  # Ensure that everything is written to disk


def log_on_batch_end(
    writer: SummaryWriter,
    current_losses: float,
    model: ImagePredictionModel,
    step: int,
) -> None:
    """Log scalar loss and optional model-level batch metrics to TensorBoard.

    Args:
        writer: TensorBoard `SummaryWriter` instance.
        current_losses: Loss value for the current batch.
        model: Model (checked for a ``__log_batch__`` hook).
        step: Global step counter.
    """
    writer.add_scalar("0_loss", current_losses, step)
    if hasattr(model, "__log_batch__"):
        model.__log_batch__(writer, step)


def tensorboard_to_hdf5(tb_dir: Union[str, pathlib.Path], out_filepath: Union[str, pathlib.Path]) -> None:
    """Convert TensorBoard event files to a single HDF5 archive.

    Reads scalars, histograms, text tensors, and images from the TensorBoard
    log directory and writes them into a structured HDF5 file for archival
    and post-hoc analysis.

    Args:
        tb_dir: Path to the directory containing TensorBoard event files.
        out_filepath: Destination HDF5 file path.
    """
    tb_dir = pathlib.Path(tb_dir)
    out_file = pathlib.Path(out_filepath)

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()

    def deduplicate_by_wall_time(events: Any) -> list:  # noqa: ANN401
        unique = {}
        for e in events:
            unique[e.wall_time] = e
        return sorted(unique.values(), key=lambda e: e.wall_time)

    with h5py.File(out_file, "w") as f:
        # ------------------ Scalars ------------------
        scalars_grp = f.create_group("scalars")
        for tag in ea.Tags().get("scalars", []):
            events = deduplicate_by_wall_time(ea.Scalars(tag))
            if not events:
                continue

            values = np.array([e.value for e in events])
            steps = np.array([e.step for e in events])
            wall_time = np.array([e.wall_time for e in events])

            tag_grp = scalars_grp.create_group(tag)
            tag_grp.create_dataset("value", data=values)
            tag_grp.create_dataset("step", data=steps)
            tag_grp.create_dataset("wall_time", data=wall_time)

        # ------------------ Histograms ------------------
        hist_grp = f.create_group("histograms")
        for tag in ea.Tags().get("histograms", []):
            events = ea.Histograms(tag)  # Do NOT deduplicate; keep all events
            if not events:
                continue

            # Prepare each attribute as a stacked array
            # TODO fix: only a single histogram is outputted for some reason
            tag_grp = hist_grp.create_group(tag)
            tag_grp.create_dataset("min", data=np.array([e.histogram_value.min for e in events]))
            tag_grp.create_dataset("max", data=np.array([e.histogram_value.max for e in events]))
            tag_grp.create_dataset("sum", data=np.array([e.histogram_value.sum for e in events]))
            tag_grp.create_dataset("num", data=np.array([e.histogram_value.num for e in events]))
            tag_grp.create_dataset("sum_squares", data=np.array([e.histogram_value.sum_squares for e in events]))
            tag_grp.create_dataset("bucket", data=np.stack([np.array(e.histogram_value.bucket) for e in events]))
            tag_grp.create_dataset(
                "bucket_limit", data=np.stack([np.array(e.histogram_value.bucket_limit) for e in events])
            )
            tag_grp.create_dataset("step", data=np.array([e.step for e in events]))
            tag_grp.create_dataset("wall_time", data=np.array([e.wall_time for e in events]))

        # ------------------ Text ------------------
        text_grp = f.create_group("text")
        for tag in ea.Tags().get("tensors", []):
            if not tag.endswith("text_summary"):
                continue
            events = deduplicate_by_wall_time(ea.Tensors(tag))
            if not events:
                continue
            serialized = [e.tensor_proto.SerializeToString() for e in events]
            tag_grp = text_grp.create_group(tag)
            tag_grp.create_dataset("data", data=np.array(serialized, dtype="S"))

        # ------------------ Images ------------------
        img_grp = f.create_group("images")
        for tag in ea.Tags().get("images", []):
            events = deduplicate_by_wall_time(ea.Images(tag))
            if not events:
                continue

            imgs = []
            for e in events:
                img = Image.open(io.BytesIO(e.encoded_image_string))
                imgs.append(np.array(img))

            imgs = np.stack(imgs)  # stack along first axis
            tag_grp = img_grp.create_group(tag)
            tag_grp.create_dataset("data", data=imgs)
