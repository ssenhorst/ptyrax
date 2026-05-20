import functools
import logging
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, NamedTuple, Union

import chex
import equinox as eqx
import gin
import jax.debug
import jax.numpy as jnp
import optax
from jax import jit, lax, vmap
from jaxtyping import Array, ArrayLike, Bool, Complex, Float, Integer, PyTree

from ptyrax.initializers import (
    support,
)
from ptyrax.models.ptychography import (
    CoherentField,
    ImagePredictionModel,
    NamedLoss,
    PtychographyModel,
)
from ptyrax.utils import (
    abs_sq,
    make_path_string,
    plot,
)

# region Optimizers


@dataclass
class OptimizerSpecification:
    """Groups an optimizer, learning-rate schedule, and parameter-matching
    patterns.

    Instances are created by :py:func:`make_optimizer_specification` and consumed by
    :py:func:`initialize_optimizer_and_state` to partition model parameters into
    optimizer groups.

    Attributes:
        name: Human-readable label (usually the gin scope name).
        match_patterns: Regex patterns that select which parameter paths use this optimizer.
        optimizer: The composed `optax.GradientTransformation`.
        learn_rate_schedule: Learning-rate schedule applied to the optimizer.
    """

    name: str
    match_patterns: list[str]
    optimizer: optax.GradientTransformation
    learn_rate_schedule: optax.Schedule = optax.constant_schedule(1e-4)


@gin.configurable
def make_optimizer_specification(
    base_optimizer: Callable[[optax.ScalarOrSchedule], optax.GradientTransformation] | optax.GradientTransformation,
    match_patterns: list[str],
    learn_rate: float = None,
    learn_rate_schedule: optax.Schedule = None,
    optimizer_pre_transforms: tuple[optax.GradientTransformation, ...] = (),
    optimizer_wrappers: tuple[Callable[[optax.GradientTransformation], optax.GradientTransformation], ...] = (),
    optimizer_post_transforms: tuple[optax.GradientTransformation, ...] = (),
    schedule_wrappers: tuple[Callable[[optax.Schedule], optax.Schedule], ...] = (),
    **kwargs,
) -> OptimizerSpecification:
    """Construct an `OptimizerSpecification` grouping optimizer, learning-rate
    schedule, and matching patterns.

    This factory accepts either an `optax.GradientTransformation` or a callable that builds one from a
    learning rate. It supports wrapping transforms and schedule wrappers, and records the gin scope name
    as the specification `name`.

    Args:
        base_optimizer: Optimizer factory or ready `GradientTransformation`.
        match_patterns: List of regex patterns selecting parameters for this optimizer.
        learn_rate: Constant learning rate (mutually exclusive with `learn_rate_schedule`).
        learn_rate_schedule: optax schedule to use for this optimizer.

    Returns:
        `OptimizerSpecification` describing optimizer and schedule for partitioning.
    """
    if learn_rate is None and learn_rate_schedule is None:
        raise ValueError("Either learn_rate or learn_rate_schedule must be provided. Both are None.")
    if learn_rate is not None and learn_rate_schedule is not None:
        raise ValueError("Either learn_rate or learn_rate_schedule must be provided. Both are set.")

    # Name follows last gin scope
    try:
        name = gin.current_scope()[-1]
    except IndexError:  # No scope: only global config
        name = "__main__"

    # Prepare optimizer
    if not isinstance(base_optimizer, optax.GradientTransformation):
        optimizer = base_optimizer(learn_rate, **kwargs)
        learn_rate = 1.0  # Learning rate is baked into the optimizer
    else:
        optimizer = base_optimizer

    for wrapper in optimizer_wrappers:
        optimizer = wrapper(optimizer)

    # Prepare learning rate schedule
    if learn_rate_schedule is None:
        learn_rate_schedule = optax.constant_schedule(learn_rate)

    learn_rate_schedule = _scale_schedule_steps_by_epoch(learn_rate_schedule)
    for wrapper in schedule_wrappers:
        learn_rate_schedule = wrapper(learn_rate_schedule)

    scale_step = optax.scale_by_schedule(learn_rate_schedule)

    final_optimizer = optax.chain(
        *optimizer_post_transforms,
        optimizer,
        scale_step,
        *optimizer_pre_transforms,
        optax.zero_nans(),
        optax.scale(-1),  # Gradient descent step
    )

    return OptimizerSpecification(
        name=name,
        match_patterns=match_patterns,
        optimizer=final_optimizer,
        learn_rate_schedule=learn_rate_schedule,
    )


@gin.configurable
def initialize_optimizer_and_state(
    model: PtychographyModel,
    *,
    optimizers: list[OptimizerSpecification] | None = None,
) -> tuple[PyTree[PtychographyModel], optax.GradientTransformation]:
    """Initialize partitioned optimizers and their state for a model based on
    ``OptimizerSpecification`` objects.

    Args:
        model: The model whose parameters will be partitioned and optimized.
        optimizers: List of `OptimizerSpecification` objects describing optimizer assignments.

    Returns:
        A tuple of (optimizer_state, partitioned_optimizer, dynamic_variable_spec).
    """
    if optimizers is None:
        warnings.warn(
            "No optimizer was configured, using default optimizer (all parameters off.). "
            "This is not recommended. Check your configuration file...",
            UserWarning,
        )
        optimizers = []

    def make_label_pairs_from_specifications(
        optimizers: list[OptimizerSpecification],
    ) -> list[tuple[str, str]]:
        label_pairs = []
        for opt_spec in optimizers:
            for pattern in opt_spec.match_patterns:
                label_pairs.append((pattern, opt_spec.name))
        return label_pairs

    if "off" not in [opt_spec.name for opt_spec in optimizers]:
        optimizers.append(
            OptimizerSpecification(
                name="off",
                match_patterns=[r".*"],
                optimizer=optax.set_to_zero(),
                learn_rate_schedule=optax.constant_schedule(0.0),
            )
        )

    def _parameter_labeler(path: list, _leaf, label_pairs: list[tuple[str, str]] = ()) -> str:
        path_string = make_path_string(path)
        if not eqx.is_array(_leaf):
            logging.info(f"{path_string} [not an array, always off]: off")
            return "off"
        for pattern, label in label_pairs:
            if re.match(pattern, path_string):
                logging.debug(f"{path_string}: {label}")
                return label

        logging.debug(f"{path_string}: off")
        return "off"

    # Label pairs are tuples of (regex_pattern, label), where regex_pattern is used to match parameter paths
    # and label matches the gin config scope of the function that generates the optimizer and schedule.
    label_pairs = make_label_pairs_from_specifications(optimizers)

    optimizer_mapping = {opt_spec.name: opt_spec.optimizer for opt_spec in optimizers}
    schedule_label_tree = jax.tree.map_with_path(functools.partial(_parameter_labeler, label_pairs=label_pairs), model)

    partitioned_optimizer = optax.transforms.partition(
        optimizer_mapping,
        # We cannot pass label_tree directly, since optax.transform will treat it as a callable.
        # This is a hack that prevents .partition() from calling the PtychographyModel
        lambda _: schedule_label_tree,
    )
    optimizer_state = partitioned_optimizer.init(model)
    return optimizer_state, partitioned_optimizer


def _tree_update_moment(updates: PyTree, moments: PyTree, decay: float, order: int, where: PyTree = None) -> PyTree:
    """Compute the exponential moving average of the `order`-th moment."""

    def moment_fn(g, t, w):  # noqa: ANN001
        if g is None:
            return None
        update = (1 - decay) * (g**order) + decay * t
        if w is not None:
            update = jax.lax.select(w, update, t)

    return jax.tree.map(
        moment_fn,
        updates,
        moments,
        where,
        is_leaf=lambda x: x is None,
    )


def _tree_update_moment_per_elem_norm(
    updates: PyTree, moments: PyTree, decay: float, order: int, where: PyTree = None
) -> PyTree:
    """Compute the EMA of the `order`-th moment of the element-wise norm."""

    def orderth_norm(g):  # noqa: ANN001
        if jnp.isrealobj(g):
            return g**order

        half_order = order / 2
        # JAX generates different HLO for int and float `order`
        if half_order.is_integer():
            half_order = int(half_order)
        return abs_sq(g) ** half_order

    def moment_fn(g, t, w):  # noqa: ANN001
        if g is None:
            return None
        update = (1 - decay) * orderth_norm(g) + decay * t
        if w is not None:
            update = jax.lax.select(w, update, t)

    return jax.tree.map(
        moment_fn,
        updates,
        moments,
        where,
        is_leaf=lambda x: x is None,
    )


@functools.partial(jax.jit, inline=True)
def _tree_bias_correction(moment: PyTree, decay: float, count: PyTree, where: PyTree = None) -> PyTree:
    """Performs bias correction.

    It becomes a no-op as count goes to infinity.
    """

    # The conversion to the data type of the moment ensures that bfloat16 remains
    # bfloat16 in the optimizer state. This conversion has to be done after
    # `bias_correction_` is calculated as calculating `decay**count` in low
    # precision can result in it being rounded to 1 and subsequently a
    # "division by zero" error.
    def bias_fn(c):  # noqa: ANN001
        return None if c is None else 1 - decay**c

    bias_correction_ = jax.tree.map(bias_fn, count, is_leaf=lambda x: x is None)

    def corr_fn(t, b, w):  # noqa: ANN001
        if t is None:
            return None
        if w is not None:
            b = jax.lax.select(w, b, jnp.array(1.0, b.dtype))
        return (t / b).astype(t.dtype)

    return jax.tree.map(corr_fn, moment, bias_correction_, where, is_leaf=lambda x: x is None)


class ThresholdedScaleByAdamState(NamedTuple):
    """State for the Adam algorithm."""

    count: PyTree  # Step count: Per parameter!
    mu: PyTree  # First moment.
    nu: PyTree  # Second moment.


def _tree_safe_increment(count: PyTree, where: PyTree = None) -> chex.Numeric:
    """Increments counter by one while avoiding overflow.

    Denote ``max_val``, ``min_val`` as the maximum, minimum, possible values for
    the ``dtype`` of ``count``. Normally ``max_val + 1`` would overflow to
    ``min_val``. This functions ensures that when ``max_val`` is reached the
    counter stays at ``max_val``.

    Args:
      count: a counter to be incremented.

    Returns:
      A counter incremented by 1, or ``max_val`` if the maximum value is
      reached.

    Examples:
      >>> import jax.numpy as jnp
      >>> import optax
      >>> optax.safe_increment(jnp.asarray(1, dtype=jnp.int32))
      Array(2, dtype=int32)
      >>> optax.safe_increment(jnp.asarray(2147483647, dtype=jnp.int32))
      Array(2147483647, dtype=int32)

    .. versionadded:: 0.2.4
    """

    def _safe_increment(count: chex.Numeric, where: Bool[Array, ""] = None) -> chex.Numeric:
        if count is None:
            return None
        count_dtype = jnp.asarray(count).dtype
        if jnp.issubdtype(count_dtype, jnp.integer):
            max_value = jnp.iinfo(count_dtype).max
        elif jnp.issubdtype(count_dtype, jnp.floating):
            max_value = jnp.finfo(count_dtype).max
        else:
            raise ValueError(
                f"Cannot safely increment count with dtype {count_dtype},"
                ' valid dtypes are subdtypes of "jnp.integer" or "jnp.floating".'
            )
        max_value = jnp.array(max_value, count_dtype)
        one = jnp.array(1, count_dtype)
        if where is not None:
            return jnp.where(jnp.logical_and(count < max_value, where), count + one, max_value)
        else:
            return jnp.where(count < max_value, count + one, max_value)

    return jax.tree.map(_safe_increment, count, where, is_leaf=lambda x: x is None)


@gin.configurable
def _scale_by_adam_thresholded(
    threshold: float = 1e-8,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Any = None,  # noqa: ANN401
    *,
    nesterov: bool = False,
) -> optax.GradientTransformation:
    """Modified directly from the optax scale_by_adam implementation."""
    mu_dtype = jax.dtypes.canonicalize_dtype(mu_dtype) if mu_dtype is not None else None

    def init_fn(params: PyTree) -> optax.ScaleByAdamState:
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
        nu = optax.tree.zeros_like(params)  # Second moment
        count = optax.tree.zeros_like(params, dtype=jnp.int32)  # Step count: Per parameter!
        return ThresholdedScaleByAdamState(count=count, mu=mu, nu=nu)

    def update_fn(
        updates: PyTree,
        state: optax.ScaleByAdamState,
        params: Any = None,  # noqa: ANN401
    ) -> tuple[PyTree, optax.ScaleByAdamState]:
        where = jax.tree.map(
            lambda u: abs_sq(u) > threshold**2 if u is not None else None, updates, is_leaf=lambda x: x is None
        )
        del params
        mu = _tree_update_moment(updates, state.mu, b1, 1, where=where)
        nu = _tree_update_moment_per_elem_norm(updates, state.nu, b2, 2, where=where)
        count_inc = _tree_safe_increment(state.count, where=where)
        if nesterov:
            mu_hat = jax.tree.map(
                lambda m, g: b1 * m + (1 - b1) * g,
                _tree_bias_correction(mu, b1, optax.numerics.safe_increment(count_inc), where=where),
                _tree_bias_correction(updates, b1, count_inc, where=where),
            )
        else:
            mu_hat = _tree_bias_correction(mu, b1, count_inc, where=where)
        # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
        # unclear why. Other Nadam implementations also omit the extra b2 factor.
        nu_hat = _tree_bias_correction(nu, b2, count_inc, where=where)

        def sparse_adam_update(m, v, u, where):  # noqa: ANN001
            if m is None:
                return None
            if where is not None:
                return jax.lax.select(where, m / (jnp.sqrt(v + eps_root) + eps), u)
            return m / (jnp.sqrt(v + eps_root) + eps)

        updates = jax.tree.map(
            sparse_adam_update,
            mu_hat,
            nu_hat,
            updates,
            where,
            is_leaf=lambda x: x is None,
        )
        mu = optax.tree.cast(mu, mu_dtype)
        return updates, ThresholdedScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


@gin.configurable
def adam_thresholded(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    threshold: float = 1e-8,
    mu_dtype: Any | None = None,  # noqa: ANN401
    *,
    nesterov: bool = False,
) -> optax.GradientTransformation:
    """Adam optimizer factory with thresholding to ignore very small updates.

    Args:
        learning_rate: Scalar or schedule for the learning rate.
        b1, b2, eps, eps_root: Adam hyperparameters.
        threshold: Minimum update magnitude to consider.
        mu_dtype: Optional dtype for first-moment accumulators.
        nesterov: Whether to use Nesterov-style correction.

    Returns:
        An `optax.GradientTransformation` implementing the thresholded Adam update.
    """
    return optax.chain(
        _scale_by_adam_thresholded(
            threshold=threshold,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        optax.scale_by_learning_rate(learning_rate),
    )


# endregion
# region Learning rate schedules
def schedule_product(
    *schedules: Iterable[optax.Schedule],
) -> optax.Schedule:
    """Combines multiple schedules by multiplying their outputs.

    Args:
        *schedules: The schedules to combine.

    Returns:
        A schedule that is the product of the input schedules.
    """

    def schedule(step: int) -> float:
        result = 1.0
        for sched in schedules:
            result *= sched(step)
        return result

    return schedule


@gin.configurable
def loop_schedule(
    schedule: optax.Schedule,
    schedule_duration: int | float,
) -> optax.Schedule:
    """Wraps a schedule so that it restarts after every ``schedule_duration``
    steps.

    Args:
        schedule (optax.Schedule): The base schedule to loop.
        schedule_duration (int | float): Number of steps before the schedule resets.

    Returns:
        optax.Schedule: A new schedule that cycles the base schedule.

    Example:
        ```python
        cosine = optax.cosine_decay_schedule(1e-3, 100)
        looped = loop_schedule(cosine, schedule_duration=100)
        ```
    """

    def wrapped_schedule(count: float) -> float:
        return schedule(count % schedule_duration)

    return wrapped_schedule


@gin.configurable
def _scale_schedule_steps_by_epoch(
    schedule: optax.Schedule,
    *,
    epoch_size: int = 1,
) -> optax.Schedule:
    """Scales the gradients by the epoch size.

    Args:
        epoch_size (int): The size of the epoch.

    Returns:
        optax.GradientTransformation: The gradient transformation that scales the gradients.
    """

    def scaled_schedule(step: float) -> float:
        return schedule(step / epoch_size)

    return scaled_schedule


# endregion
# region Loss functions


@gin.configurable()
def mean_square_error(y_true: Float[Array, ""], y_pred: Float[Array, ""]) -> Float[Array, ""]:
    """Computes the mean square error between the true and predicted values.

    Args:
        y_true (array): The true values.
        y_pred (array): The predicted values.

    Returns:
        float: The computed mean square error.
    """
    return jnp.mean((y_true - y_pred) ** 2)


@jit
@gin.configurable
def loss(
    model: ImagePredictionModel,
    data_indices: Integer[Array, ""],
    target_image: Float[Array, ""],
    loss_fn: Callable[[Float[Array, ""], Float[Array, ""]], float] = mean_square_error,
    batch_reduction_fn: Callable[[jnp.ndarray], float] = jnp.sum,
) -> tuple[float, tuple[list[NamedLoss], Float[Array, " N"]]]:
    """Computes the loss for the model.

    Args:
        model (PtychographyModel): The model to compute the loss for.
        scanning_position_index (array): The scanning position index.
        target_diffraction_pattern (array): The target diffraction pattern.
        loss_fn (Callable): The loss function to use.

    Returns:
        float: The computed loss.
    """
    # From here on, everything that should be an array, actually is an array.

    def call_model(data_indices: Integer[Array, " batch"]) -> Float[Array, " N w h"]:
        model_resolved = model.resolve(data_indices)
        # TODO move this into ImagePredictionModel?
        return model_resolved()

    detected_field = vmap(call_model)(data_indices)
    losses = []
    loss_per_index = vmap(loss_fn)(target_image, detected_field)
    loss_total = batch_reduction_fn(loss_per_index)
    losses.append(NamedLoss("Data_fidelity", loss_per_index))
    return loss_total, losses


# THIS FUNCTION WAS PREVIOUSLY UNSTABLE, RETURNING NaN VALUES EVEN FOR NON-TRACED
# Y_TRUE. BE CAREFUL!
# @gin.configurable()
# def modulus_mean_square_error(
#   y_true, y_pred, rect_fun=lambda y: jnp.where(y>0, y, 0.)
# ):
#     y_true = rect_fun(jnp.abs(y_true))
#     y_pred = rect_fun(jnp.abs(y_pred))
#     return jnp.mean((jnp.sqrt(y_true) - jnp.sqrt(y_pred)) ** 2)


@gin.configurable()
def threshold_relative_mean_square_error(
    y_true: Float[Array, ""],
    y_pred: Float[Array, ""],
    eps: float = 0.01,
    threshold: float = 0.0004,
) -> Float[Array, ""]:
    """Computes the relative mean square error between the true and predicted
    values.

    Args:
        y_true (array): The true values.
        y_pred (array): The predicted values.
        eps (float): A small value to avoid division by zero.

    Returns:
        float: The computed relative mean square error.
    """
    y_true = jnp.sqrt(jax.nn.relu(y_true**2 - threshold))
    y_pred = jnp.sqrt(jax.nn.relu(y_pred**2 - threshold))
    return jnp.sum((y_true - y_pred) ** 2) / jnp.sum(y_true**2)


@gin.configurable()
def relative_mean_square_error(
    y_true: Float[Array, ""],
    y_pred: Float[Array, ""],
    eps: float = 0.01,
) -> Float[Array, ""]:
    """Computes the relative mean square error between the true and predicted
    values.

    Args:
        y_true (array): The true values.
        y_pred (array): The predicted values.
        eps (float): A small value to avoid division by zero.

    Returns:
        float: The computed relative mean square error.
    """
    return jnp.sum((y_true - y_pred) ** 2) / jnp.sum(y_true**2)


@gin.configurable()
def relative_mean_square_error_per_pixel(
    y_true: Float[Array, ""],
    y_pred: Float[Array, ""],
    eps: float = 1e-8,
) -> Float[Array, ""]:
    """Computes the relative mean square error between the true and predicted
    values on a per-pixel basis.

    Args:
        y_true (array): The true values.
        y_pred (array): The predicted values.
        eps (float): A small value to avoid division by zero.

    Returns:
        float: The computed relative mean square error.
    """
    y_pred **= 2
    y_true **= 2
    return jnp.mean(((y_true - y_pred) / (lax.stop_gradient(y_pred) + eps)) ** 2)


@gin.configurable()
def mixed_mean_square_error(
    y_true: Float[Array, ""],
    y_pred: Float[Array, ""],
    eps: float = 0.01,
) -> Float[Array, ""]:
    """Computes the relative mean square error between the true and predicted
    values.

    Args:
        y_true (array): The true values.
        y_pred (array): The predicted values.
        eps (float): A small value to avoid division by zero.

    Returns:
        float: The computed relative mean square error.
    """
    # return jnp.sum((y_true - y_pred) / (lax.stop_gradient(y_pred) + eps))
    return jnp.sum((y_true - y_pred) ** 2 / (jnp.sqrt((y_true + eps) * (lax.stop_gradient(y_pred) + eps))))


def mean_error(
    y_true: Float[Array, ""],
    y_pred: Float[Array, ""],
) -> Float[Array, ""]:
    """Computes the mean error between the true and predicted values.

    Args:
        y_true (array): The true values.
        y_pred (array): The predicted values.

    Returns:
        float: The computed mean error.
    """
    return jnp.mean(y_true - y_pred)


# endregion

# region Regularization functions


@gin.configurable
def regularize(
    model: ImagePredictionModel,
) -> tuple[float, list[NamedLoss]]:
    """Computes the regularization term for the model.

    Args:
        model (ImagePredictionModel): The model to regularize.

    Returns:
        float: The computed regularization term.
    """
    regularization_total, regularization_terms = model.__regularize__()
    regularization_terms.append(NamedLoss("Total_regularization", regularization_total))
    return regularization_total, regularization_terms


@jit
@gin.configurable
def support_overlap(field: CoherentField, weight: float = 0.0, **kwargs) -> float:
    """Computes the support overlap regularization term.

    Args:
        field (CoherentField): The field to compute the support overlap for.
        weight (float): The weight of the regularization term.

    Returns:
        float: The computed support overlap regularization term.
    """
    support_matrix = support(field.sampling.meshgrid, **kwargs)
    out = jnp.sum(jnp.abs(field()) * jnp.abs(support_matrix)) / jnp.sqrt(
        jnp.sum(jnp.abs(field()) ** 2) * jnp.sum(support_matrix * jnp.conj(support_matrix))
    )
    return jnp.abs(out) * weight


@gin.configurable()
def support_l2(field: CoherentField, weight: float = 0.0, **kwargs) -> float:
    """Computes the L2 norm of the field within a support region.

    Multiplies the field data by a support mask and returns the weighted L2 norm.
    The support mask is generated from the field's sampling grid using
    :py:func:`~ptyrax.initializers.support`.

    Args:
        field (CoherentField): The coherent field whose data is masked and normed.
        weight (float): Scalar multiplier for the regularization term. Defaults to 0.0.
        **kwargs: Additional keyword arguments forwarded to
            :py:func:`~ptyrax.initializers.support`.

    Returns:
        float: The weighted L2 norm of the support-masked field.

    Example:
        ```python
        reg = support_l2(probe_field, weight=1e-3, radius=0.5)
        ```
    """
    support_matrix = support(field.sampling.meshgrid, **kwargs)
    return _l2(support_matrix[..., jnp.newaxis] * field.data, weight)


@gin.configurable()
def support_l1(field: CoherentField, weight: float = 0.0, **kwargs) -> float:
    """Computes the L1 norm of the support matrix.

    Args:
        field (CoherentField): The field to compute the L1 norm for.
        weight (float): The weight of the regularization term.

    Returns:
        float: The computed L1 norm of the support matrix.
    """
    support_matrix = support(field.sampling.meshgrid, **kwargs)
    jax.debug.callback(lambda sup: plot(sup, show=True), support_matrix)
    return _l1(support_matrix[..., jnp.newaxis] * field.data, weight)


@gin.configurable
def tv(field: CoherentField, weight: float = 0.0, tv_mode: str = "real_imag") -> float:
    """Computes the total variation regularization term.

    Args:
        field (CoherentField): The field to compute the total variation for.
        weight (float): The weight of the regularization term.
        tv_mode (str): The mode of total variation computation. Can be "real_imag" or "mag_phase".

    Returns:
        float: The computed total variation regularization term.

    Raises:
        ValueError: If the tv_mode is not one of "real_imag" or "mag_phase".
    """
    data = field.data
    if tv_mode == "real_imag":
        regularization = _tv(jnp.real(data), weight)
        regularization += _tv(jnp.imag(data), weight)
        return regularization
    elif tv_mode == "mag_phase":
        regularization = _tv(jnp.abs(data), weight)
        regularization += _tv(jnp.angle(data), weight)
        return regularization
    else:
        raise ValueError(
            f"Total variation specification not clear. Expected one of ('real_imag, mag_phase'),got {tv_mode}"
        )


@gin.configurable
def _tv(a: Complex[Array, ""], weight: float) -> float:
    """Helper function to compute the total variation of an array.

    Args:
        a (array): The array to compute the total variation for.
        weight (float): The weight of the regularization term.

    Returns:
        float: The computed total variation.
    """
    if len(a.shape) < 2:
        raise ValueError("Input array must be at least 2D for total variation computation.")
    return _l1(jnp.abs(a[:-1, :] - a[1:, :]), 1) * weight + _l1(jnp.abs(a[:, :-1] - a[:, 1:]), 1) * weight


def bases_l1(field: CoherentField, weight: float = 0.0) -> Array:
    """Computes the L1 norm of the base arrays used to compute the field
    samples. For example, if data is modeled by an
    OuterProductArrayParametrization, the arrays present in the outer product
    are used to compute the l1 norm. Mostly useful if wavelet bases are used,
    as natural images are expected to have sparse l1 in these bases.

    Args:
        field (CoherentField): The field to compute the L1 norm for.
        weight (float): The weight of the regularization term.

    Returns:
        float: The computed L1 norm of the bases.
    """
    return jnp.sum(jax.tree_map(_l1, field))


@gin.register
def parameter_l1(pytree: PyTree, weight: float = 0.0) -> Float[Array, ""]:
    """Computes the L1 norm of the parameters.

    Args:
        pytree: The pytree to compute the L1 norm for.
        weight (float): The weight of the regularization term.

    Returns:
        float: The computed L1 norm of the parameters.
    """
    return jnp.sum(jax.tree.map(_l1, pytree, weight))


@gin.configurable()
def l1(field: CoherentField, weight: float = 0.0) -> Float[Array, ""]:
    """Computes the L1 norm regularization term.

    Args:
        field (CoherentField): The field to compute the L1 norm for.
        weight (float): The weight of the regularization term.

    Returns:
        float: The computed L1 norm regularization term.
    """
    return _l1(field.data, weight)


def _norm(a: Complex[Array, ""], weight: float, order: int = 1) -> Float[Array, ""]:
    """Generic weighted norm helper.

    Args:
        a: The array to compute the norm for.
        weight: Scalar multiplier for the regularization term.
        order: Norm order (1 for L1, 2 for L2).

    Returns:
        The weighted norm.
    """
    return jnp.mean(jnp.abs(a) ** order) * weight


def _l1(a: Complex[Array, ""], weight: float) -> Float[Array, ""]:
    """Helper function to compute the L1 norm of an array.

    Args:
        a (array): The array to compute the L1 norm for.
        weight (float): The weight of the regularization term.

    Returns:
        float: The computed L1 norm.
    """
    return _norm(a, weight, order=1)


@gin.configurable()
def fft_l1(field_or_array: Union[CoherentField, ArrayLike], weight: float) -> Float[Array, ""]:
    """Computes the L1 norm of the FFT of the field data.

    Args:
        field (CoherentField): The field to compute the FFT L1 norm for.
        weight (float): The weight of the regularization term.

    Returns:
        float: The computed FFT L1 norm regularization term.
    """
    if isinstance(field_or_array, CoherentField):
        return fft_l1(field_or_array.data, weight)
    return _l1(jnp.fft.fft2(field_or_array), weight)


@gin.configurable()
def fft_l2(field: CoherentField, weight: float) -> Float[Array, ""]:
    """Computes the L2 norm of the 2-D FFT of the field data.

    Applies a 2-D FFT to `field.data` and returns the weighted L2 norm,
    encouraging low energy content in the Fourier domain.

    Args:
        field (CoherentField): The coherent field whose Fourier-space L2 norm is computed.
        weight (float): Scalar multiplier for the regularization term.

    Returns:
        Float[Array, ""]: The weighted L2 norm of the FFT of the field data.

    Example:
        ```python
        reg = fft_l2(probe_field, weight=1e-4)
        ```
    """
    return _l2(jnp.fft.fft2(field.data), weight)


@gin.configurable()
def l2(field: CoherentField, weight: float = 0.0) -> Float[Array, ""]:
    """Computes the L2 norm regularization term.

    Args:
        field (CoherentField): The field to compute the L2 norm for.
        weight (float): The weight of the regularization term.

    Returns:
        float: The computed L2 norm regularization term.
    """
    return _l2(field.data, weight)


@gin.configurable()
def _l2(a: Complex[Array, ""], weight: float) -> Float[Array, ""]:
    """Helper function to compute the L2 norm of an array.

    Args:
        a (array): The array to compute the L2 norm for.
        weight (float): The weight of the regularization term.

    Returns:
        float: The computed L2 norm.
    """
    return _norm(a, weight, order=2)


# endregion
