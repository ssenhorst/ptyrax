import logging

import gin
import optax

_logger = logging.getLogger(__name__)


def _safe_register(names: tuple[str, ...], module: str = "optax") -> None:
    """Register optax attributes as gin configurables, skipping missing
    ones."""
    for name in names:
        attr = getattr(optax, name, None)
        if attr is None:
            _logger.warning("optax.%s not found in optax %s — skipping gin registration", name, optax.__version__)
            continue
        gin.external_configurable(attr, module=module)


# Optax configurables
## Optimizers
_safe_register(
    (
        "adabelief",
        "adadelta",
        "adafactor",
        "adagrad",
        "adam",
        "adamax",
        "adamaxw",
        "adamw",
        "adan",
        "amsgrad",
        "fromage",
        "lamb",
        "lars",
        "lbfgs",
        "lion",
        "noisy_sgd",
        "novograd",
        "optimistic_gradient_descent",
        "optimistic_adam_v2",
        "polyak_sgd",
        "radam",
        "rmsprop",
        "rprop",
        "sgd",
        "sign_sgd",
        "sm3",
        "yogi",
    )
)

## Optimizer transformations
_safe_register(
    (
        "adaptive_grad_clip",
        "add_decayed_weights",
        "add_noise",
        "apply_every",
        "bias_correction",
        "conditionally_mask",
        "conditionally_transform",
        "centralize",
        "clip",
        "clip_by_block_rms",
        "clip_by_global_norm",
        "ema",
        "global_norm",
        "identity",
        "keep_params_nonnegative",
        "normalize_by_update_norm",
        "per_example_global_norm_clip",
        "per_example_layer_norm_clip",
        "scale",
        "scale_by_adadelta",
        "scale_by_adan",
        "scale_by_adam",
        "scale_by_adamax",
        "scale_by_amsgrad",
        "scale_by_backtracking_linesearch",
        "scale_by_belief",
        "scale_by_factored_rms",
        "scale_by_lbfgs",
        "scale_by_lion",
        "scale_by_novograd",
        "scale_by_optimistic_gradient",
        "scale_by_param_block_norm",
        "scale_by_param_block_rms",
        "scale_by_polyak",
        "scale_by_rms",
        "scale_by_rprop",
        "scale_by_rss",
        "scale_by_schedule",
        "scale_by_sign",
        "scale_by_sm3",
        "scale_by_stddev",
        "scale_by_trust_ratio",
        "scale_by_yogi",
        "scale_by_zoom_linesearch",
        "set_to_zero",
    )
)

## Combiners
_safe_register(("chain", "named_chain", "partition"))

## Wrappers
_safe_register(
    (
        "apply_if_finite",
        "flatten",
        "lookahead",
        "masked",
        "MultiSteps",
        "skip_large_updates",
        "skip_not_finite",
    )
)

# Schedules
_safe_register(
    (
        "constant_schedule",
        "cosine_decay_schedule",
        "cosine_onecycle_schedule",
        "exponential_decay",
        "join_schedules",
        "linear_onecycle_schedule",
        "linear_schedule",
        "piecewise_constant_schedule",
        "piecewise_interpolate_schedule",
        "polynomial_schedule",
        "sgdr_schedule",
    )
)

# Losses
_safe_register(
    (
        "convex_kl_divergence",
        "cosine_distance",
        "cosine_similarity",
        "ctc_loss",
        "ctc_loss_with_forward_probs",
        "hinge_loss",
        "huber_loss",
        "kl_divergence",
        "l2_loss",
        "log_cosh",
        "ntxent",
        "sigmoid_focal_loss",
        "smooth_labels",
        "softmax_cross_entropy",
        "softmax_cross_entropy_with_integer_labels",
        "squared_error",
    )
)
