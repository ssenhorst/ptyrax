from abc import abstractmethod
from typing import Any, Callable, Generic, Optional, TypeVar, Union

import equinox as eqx
import gin
import jax
import jax.numpy as jnp
import jaxwt as jwt
import numpy as np
from chromatix.ops import binarize
from jaxtyping import Array, Bool, Complex, Float, PyTree, Shaped


def phase_only_exp(phase: Float[Array, "... m n"]) -> Complex[Array, "... m n"]:
    r"""Convert a real-valued phase array to a complex unit-magnitude array.

    Computes :math:`e^{i \phi}` as :math:`\cos(\phi) + i \sin(\phi)` using JAX.

    Args:
        phase: Real-valued array of phases in radians.

    Returns:
        Complex array with unit magnitude and the given phases.

    Raises:
        ValueError: If the input array is complex-valued.
    """
    if jnp.iscomplexobj(phase):
        raise ValueError("Input to phase_only_exp should be real-valued phase array.")
    return jnp.cos(phase) + 1j * jnp.sin(phase)


def phase_only_exp_np(phase: Float[Array, "... m n"]) -> Complex[Array, "... m n"]:
    r"""Convert a real-valued phase array to a complex unit-magnitude array
    using NumPy.

    Computes :math:`e^{i \phi}` as :math:`\cos(\phi) + i \sin(\phi)` using NumPy
    (host-side, non-JIT-compatible).

    Args:
        phase: Real-valued NumPy array of phases in radians.

    Returns:
        Complex NumPy array with unit magnitude and the given phases.

    Raises:
        ValueError: If the input array is complex-valued.
    """
    if np.iscomplexobj(phase):
        raise ValueError("Input to phase_only_exp should be real-valued phase array.")
    return np.cos(phase) + 1j * np.sin(phase)


T = TypeVar("T")


def resolve_parametrizations(module: T, index: int | None = None) -> T:
    """Recursively resolve all parametrizations in a module to their underlying
    arrays.

    Resolves both :py:class:`~ptyrax.parametrizations.ArrayParametrization` and
    :py:class:`~ptyrax.parametrizations.IndexDependentParameter` instances in the
    module's pytree. This should be called before using a model inside a JAX
    jit-compiled function.

    Args:
        module: The module (or pytree) containing parametrizations to resolve.
        index: If provided, resolves index-dependent parameters at this specific
            dataset index. If ``None``, resolves all indices at once (stacking
            along the leading dimension).

    Returns:
        A copy of the module with all parametrizations replaced by their
        resolved array values.

    Example:
        >>> resolved_model = resolve_parametrizations(model, index=0)
        >>> output = resolved_model(inputs)
    """
    module = resolve_array_parametrizations(module)
    if index is not None:
        module = resolve_index_dependent_parameters(module, index)
    else:
        module = resolve_index_dependent_parameters_all(module)
    return module


def resolve_array_parametrizations(module: T) -> T:
    """Recursively resolve all
    :py:class:`~ptyrax.parametrizations.ArrayParametrization` instances in a
    module.

    Traverses the pytree and replaces each
    :py:class:`~ptyrax.parametrizations.ArrayParametrization` leaf with the
    array returned by calling it. Nested parametrizations are resolved
    recursively.

    Args:
        module: The module (or pytree) containing array parametrizations.

    Returns:
        A copy of the module with all array parametrizations replaced by
        their output arrays.
    """

    def resolve_fn(x: Any) -> Any:  # noqa: ANN401
        if isinstance(x, ArrayParametrization):
            # Recursion ensures nested parametrizations are also resolved
            return resolve_array_parametrizations(x())
        return x

    return jax.tree.map(resolve_fn, module, is_leaf=lambda x: isinstance(x, ArrayParametrization))  # type: ignore


class ArrayParametrization(eqx.Module):
    """Abstract base class for array parametrizations.

    An array parametrization wraps trainable parameters and produces an output
    array via its ``__call__`` method. Subclasses implement specific constraints
    (e.g., phase-only, normalized, outer-product decomposition) while exposing
    a uniform interface that can be resolved transparently using
    :py:func:`~ptyrax.parametrizations.resolve_array_parametrizations`.

    Attributes:
        output_shape: The shape of the array produced by ``__call__``.

    Example:
        >>> parametrized_model = build_model(config)
        >>> resolved = resolve_parametrizations(parametrized_model, index=0)
        >>> output = resolved(inputs)
    """

    output_shape: tuple = eqx.field(static=True)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Array:
        """Compute and return the parametrized output array."""
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the output array produced by this parametrization."""
        return self.output_shape

    def __getitem__(self, name: str) -> Any:  # noqa: ANN401
        try:
            return super().__getattribute__(name)
        except (AttributeError, TypeError) as e:
            raise AttributeError(
                f"""'ArrayParametrization' object has no item '{name}'. 
                It is likely that the module you are trying to access has not resolved parametrizations yet. 
                Please use the 'resolve_parametrizations' function to resolve all parametrizations before
                calling the module, 
                i.e. 'resolved_module = resolve_parametrizations(module, index); resolved_module(...)'.
                Alternatively, when operating outside the jit-boundary, you can access all parametrizations 
                via the 'all' property, i.e. 'all_parametrizations = module.foo.all'.
                """
            ) from e

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        try:
            return super().__getattribute__(name)
        except (AttributeError, TypeError) as e:
            raise AttributeError(
                f"""'ArrayParametrization' object has no attribute '{name}'. 
                It is likely that the module you are trying to access has not resolved parametrizations yet. 
                Inside of the jit-boundary, you should use the 'resolve_parametrizations' function to resolve all 
                parametrizations before calling the module, 
                i.e. 'resolved_module = resolve_parametrizations(module, index); resolved_module(...)'.
                Alternatively, when operating outside the jit-boundary, you can access all parametrizations 
                via the 'all' property, i.e. 'all_parametrizations = module.foo.all'.
                """
            ) from e


@gin.register
def mean_l2_norm(a: Float[Array, "* n m d"], axes: tuple[int, ...] = (-3, -2)) -> Float[Array, "* d"]:
    r"""Compute the mean L2 norm of an array along the specified axes.

    This is typically used as a ``scaling_function`` for
    :py:class:`~ptyrax.parametrizations.NormalizedArrayParametrization` to
    normalize arrays to a target mean L2 norm.

    Args:
        a: Input array to compute the norm of.
        axes: Axes along which to compute the L2 norm before averaging.

    Returns:
        The mean L2 norm, with the specified axes reduced.
    """
    return jnp.mean(jnp.linalg.norm(a, axes))


@gin.register
class NormalizedArrayParametrization(ArrayParametrization):
    r"""Parametrization that normalizes an array by a fixed scale factor.

    Stores the data divided by a scale factor and multiplies by that factor
    on evaluation. This keeps the internal parameters at unit scale during
    optimization, which can improve gradient conditioning.

    The scale factor is computed either from a provided ``scaling_function``
    applied to the initial data, or from the explicit ``scale`` argument.

    Attributes:
        output_shape: Shape of the output array.
        _data: Internal normalized data (``initial_data / scale``).
        _scale: The normalization scale factor.

    Example:
        >>> param = NormalizedArrayParametrization(probe_array, scale=1e3)
        >>> resolved_array = param()  # returns probe_array
    """

    output_shape: tuple = eqx.field(static=True)
    _data: Shaped[Array, "..."]
    _scale: float

    def __init__(
        self,
        initial_data: Float[Array, "..."],
        scale: float = 1.0,
        scaling_function: Callable[[Shaped[Array, "..."]], Shaped[Array, "..."]] = None,
    ) -> None:
        """Initialize the normalized parametrization.

        Args:
            initial_data: The array to parametrize.
            scale: Fixed scale factor to normalize by. Ignored if
                ``scaling_function`` is provided.
            scaling_function: Optional callable that computes the scale factor
                from ``initial_data``. Takes precedence over ``scale``.
        """
        if scaling_function is not None:
            self._scale = scaling_function(initial_data)
        else:
            self._scale = scale
        self._data = initial_data / self._scale
        self.output_shape = initial_data.shape

    def __call__(self) -> Shaped[Array, "..."]:
        """Return the reconstructed array (data multiplied by scale).

        Returns:
            The denormalized array.
        """
        # Stop gradient prevents training of the reference value, but can also block
        # the gradient if new parametrization are created based on previously trained
        # arrays. Only use ArrayParametrizations on the model parameters, never on
        # intermediate parameters!
        output = self._data * self._scale
        return output

    def __array__(self, dtype: TypeVar | None = None) -> Shaped[Array, "..."]:
        """Support conversion to NumPy array via ``np.asarray(param)``."""
        # noqa: ANN401
        output = self._data * self._scale
        return output.astype(dtype) if dtype is not None else output


@gin.configurable
class NormalizedReferencedArrayParametrization(ArrayParametrization):
    r"""Parametrization that stores a trainable offset relative to a fixed
    reference.

    The output is computed as ``data * scale + reference_value``, where
    ``reference_value`` is the initial data (with stopped gradients) and
    ``data`` starts at zero. This allows optimization to learn a correction
    on top of an initial estimate without modifying the reference.

    Attributes:
        output_shape: Shape of the output array.
        _data: Trainable offset initialized to zeros.
        _scale: Scale factor applied to the trainable offset.
        _reference_value: Fixed reference array (gradient-stopped).

    Example:
        >>> param = NormalizedReferencedArrayParametrization(initial_probe)
        >>> # Initially returns initial_probe (since _data is zero)
        >>> assert jnp.allclose(param(), initial_probe)
    """

    output_shape: tuple = eqx.field(static=True)
    _data: Shaped[Array, "..."]
    _scale: float
    _reference_value: Shaped[Array, "..."]

    def __init__(
        self,
        initial_data: Shaped[Array, "..."],
        scale: float = 1.0,
        scaling_function: Callable[[Shaped[Array, "..."]], Shaped[Array, "..."]] = None,
    ) -> None:
        """Initialize the normalized referenced parametrization.

        Args:
            initial_data: The initial reference array. Stored with stopped
                gradients and used as the baseline output.
            scale: Scale factor for the trainable offset.
            scaling_function: Optional callable that computes the scale factor
                from ``initial_data``. Takes precedence over ``scale``.
        """
        if scaling_function is not None:
            self._scale = scaling_function(initial_data)
        else:
            self._scale = scale
        self._reference_value = initial_data
        self._data = jnp.zeros_like(initial_data)
        self.output_shape = initial_data.shape

    def __call__(self) -> Shaped[Array, "..."]:
        """Return the parametrized array as ``data * scale + reference``.

        The reference value has its gradient stopped to prevent training
        of the initial estimate.

        Returns:
            The sum of the scaled trainable offset and the fixed reference.
        """

        # Stop gradient prevents training of the reference value, but can also block
        # the gradient if new parametrization are created based on previously trained
        # arrays. Only use ArrayParametrizations on the model parameters, never on
        # intermediate parameters!
        def broadcast_to_match(a: Shaped, b: Shaped) -> Shaped:
            # a: e.g. (N, H, W) or (H, W)
            # b: e.g. (N,) or ()
            if b.ndim == 0:
                return b  # scalar, fine
            if b.ndim == 1 and a.ndim > 1:
                # Add trailing singleton dimensions to match rank
                return b.reshape((b.shape[0],) + (1,) * (a.ndim - 1))
            return b

        scale = broadcast_to_match(self._data, jax.lax.stop_gradient(jnp.array(self._scale)))
        output = self._data * scale + jax.lax.stop_gradient(self._reference_value)
        return output


class PhaseOnlyArrayParametrization(ArrayParametrization):
    r"""Parametrization that constrains an array to unit magnitude (phase-only).

    Stores a real-valued phase array and produces a complex array with unit
    magnitude via :math:`e^{i \phi}`. This is useful for representing phase
    screens or phase-only optical elements.

    Attributes:
        output_shape: Shape of the output complex array.
        phase: Trainable real-valued phase array in radians.

    Example:
        >>> param = PhaseOnlyArrayParametrization(output_shape=(128, 128))
        >>> field = param()  # complex array with |field| == 1 everywhere
    """

    output_shape: tuple = eqx.field(static=True)
    phase: Float[Array, "M N"]

    def __init__(
        self,
        output_shape: tuple,
        phase_initializer: Callable[[tuple], Float] = jnp.zeros,
    ) -> None:
        """Initialize the phase-only parametrization.

        Args:
            output_shape: Shape of the output complex array.
            phase_initializer: Callable that takes a shape tuple and returns
                the initial phase values. Defaults to zeros.
        """
        self.phase = phase_initializer(output_shape)
        self.output_shape = output_shape

    def __call__(self, **kwargs) -> Complex[Array, ""]:
        r"""Compute the unit-magnitude complex array from the stored phase.

        Returns:
            Complex array :math:`e^{i \phi}` with unit magnitude.
        """
        return phase_only_exp(self.phase)


@gin.configurable()
class OuterProductArrayParametrization(ArrayParametrization):
    r"""Parametrization that represents a 2D array as a sum of outer products.

    Decomposes a 2D array of shape ``(M, N, d)`` into ``n_outer`` rank-1
    components, stored as column vectors of shape ``(M, n_outer, d)`` and row
    vectors of shape ``(n_outer, N, d)``. The output is reconstructed via
    Einstein summation: :math:`A_{MNd} = \sum_s c_{Msd} \cdot r_{sNd}`.

    This reduces the number of free parameters from :math:`M \times N` to
    :math:`n_{outer} \times (M + N)`, which can act as a regularizer.

    Attributes:
        output_shape: Shape of the reconstructed output array.
        column_vector: Column factors of shape ``(..., M, n_outer, d)``.
        row_vector: Row factors of shape ``(..., n_outer, N, d)``.

    Example:
        >>> param = OuterProductArrayParametrization(output_shape=(64, 64, 1), n_outer=4)
        >>> array = param()  # shape (64, 64, 1)
    """

    output_shape: tuple
    column_vector: Shaped[Array, "... M s d"]
    row_vector: Shaped[Array, "... s N d"]

    def __init__(
        self,
        output_shape: tuple,
        n_outer: int,
        initializer: Optional[Callable[[tuple], Array]] = jnp.ones,
    ) -> None:
        """Initialize the outer product parametrization.

        Args:
            output_shape: Desired shape of the output array. Must have at least
                3 dimensions ``(..., M, N, d)``.
            n_outer: Number of rank-1 outer-product components.
            initializer: Callable that takes a shape tuple and returns an
                initial array. Defaults to ``jnp.ones``.

        Raises:
            ValueError: If ``output_shape`` has fewer than 3 dimensions.
        """
        if len(output_shape) < 3:
            raise ValueError(
                "Output shape must have at least 3 dimensions to use OuterProductDataModel, "
                f"got {output_shape}. The last dimension may be 1."
            )
        self.output_shape = output_shape
        self.column_vector = initializer(output_shape[:-2] + (n_outer,) + output_shape[-1:])
        self.row_vector = initializer(output_shape[:-3] + (n_outer,) + output_shape[-2:])

    def __call__(self) -> Shaped[Array, "... M N d"]:
        """Reconstruct the full array from the outer product of column and row
        vectors.

        Returns:
            The reconstructed array of shape ``output_shape``.
        """
        return jnp.einsum(
            "...Msd, ...sNd -> ...MNd",
            self.column_vector,
            self.row_vector,
            optimize="optimal",
        )


@gin.configurable()
class WaveletArrayParametrization(ArrayParametrization):
    """Parametrization that stores an array in the wavelet domain.

    Applies a 2D Haar wavelet decomposition to the initial data and stores
    the wavelet coefficients as the trainable parameters. On evaluation,
    the inverse wavelet transform reconstructs the spatial-domain array.

    Operating in the wavelet domain can provide a multi-scale representation
    that encourages sparsity in the wavelet basis.

    Attributes:
        wavelet_coefficients: Trainable wavelet coefficients (Haar, level 1).

    Example:
        >>> param = WaveletArrayParametrization(initial_array)
        >>> reconstructed = param()  # spatial-domain array
    """

    wavelet_coefficients: Shaped[Array, "..."]

    def __init__(self, data_or_initializer: Union[Callable[[tuple], Shaped]]) -> None:
        """Initialize the wavelet parametrization.

        Args:
            data_or_initializer: Either a JAX array to decompose, or a
                zero-argument callable that returns one.
        """
        if callable(data_or_initializer):
            initial_data = data_or_initializer()
        else:
            initial_data = data_or_initializer
        self.output_shape = initial_data.shape
        self.wavelet_coefficients = jwt.wavedec2(initial_data, "haar", level=1)

    def __call__(self, *args, **kwargs) -> Shaped[Array, "..."]:
        """Reconstruct the spatial-domain array from wavelet coefficients.

        Returns:
            The inverse-wavelet-transformed array.
        """
        return jwt.waverec2(self.wavelet_coefficients, "haar")[0]


@gin.configurable()
class DirectArrayParametrization(ArrayParametrization):
    """Identity parametrization that wraps an array without any transformation.

    This is a no-op parametrization: calling it simply returns the stored data
    unchanged. It is useful for providing a uniform
    :py:class:`~ptyrax.parametrizations.ArrayParametrization` interface around
    arrays that require no constraints.

    Attributes:
        _data: The stored array.

    Example:
        >>> param = DirectArrayParametrization(jnp.ones((64, 64)))
        >>> assert jnp.array_equal(param(), jnp.ones((64, 64)))
    """

    _data: Float[Array, "..."]

    def __init__(self, data_or_initializer: Union[Callable[[tuple], Shaped]]) -> Shaped[Array, "..."]:
        """Initialize the direct parametrization.

        Args:
            data_or_initializer: Either a JAX array to store directly, or a
                zero-argument callable that returns one.
        """
        if callable(data_or_initializer):
            self._data = data_or_initializer()
        else:
            self._data = data_or_initializer

    def __call__(self) -> Float[Array, "..."]:
        """Return the stored array unchanged.

        Returns:
            The wrapped array.
        """
        return self._data

    @property
    def output_shape(self) -> tuple[int, ...]:
        """The shape of the stored array."""
        return self._data.shape


class BinaryArrayParametrization(ArrayParametrization):
    """Parametrization that produces a binary array via thresholding."""

    _data: Float[Array, "..."]
    threshold: float = eqx.field(static=True)

    def __init__(self, data_or_initializer: Union[Callable[[tuple], Shaped]], threshold: float = 0.5) -> None:
        """Initialize the binary array parametrization.

        Args:
            data_or_initializer: Either a JAX array to store, or a zero-argument
                callable that returns one.
            threshold: Threshold value for binarization. Output will be 1 where
                data > threshold and 0 elsewhere.
        """
        if callable(data_or_initializer):
            _data = data_or_initializer()
        else:
            _data = data_or_initializer
        if jnp.iscomplex(_data):
            _data = jnp.angle(_data - jnp.mean(_data))
            _data = (_data - jnp.min(_data)) / (jnp.max(_data) - jnp.min(_data))  # Normalize to [0, 1]

        self._data = _data
        self.output_shape = _data.shape
        self.threshold = threshold

    def __call__(self) -> Bool[Array, "..."]:
        """Return a binary array obtained by thresholding the stored data.

        Returns:
            Binary array where values are 1 if data > threshold and 0 otherwise.
        """
        output = binarize(self._data, self.threshold)
        return output


def as_direct_array_parametrization(data: Shaped[Array, "..."]) -> DirectArrayParametrization:
    """Wrap an array in a
    :py:class:`~ptyrax.parametrizations.DirectArrayParametrization` if needed.

    If ``data`` is already an
    :py:class:`~ptyrax.parametrizations.ArrayParametrization`, it is returned
    unchanged. Otherwise, it is wrapped in a
    :py:class:`~ptyrax.parametrizations.DirectArrayParametrization`.

    Args:
        data: An array or existing parametrization.

    Returns:
        An ``ArrayParametrization`` wrapping the input data.
    """
    if isinstance(data, ArrayParametrization):
        return data
    return DirectArrayParametrization(data)


T = TypeVar("T")


class IndexDependentParameter(eqx.Module, Generic[T]):
    """Abstract base class for parameters that vary with the dataset index.

    In ptychography, certain model parameters (e.g., scan positions, per-frame
    aberrations) differ for each diffraction pattern in the dataset. This class
    provides a uniform interface to access index-specific values either
    explicitly via :meth:`at_index` or implicitly after resolution via
    :meth:`at_current_index`.

    Subclasses must implement :meth:`at_index`, :attr:`n`, and :attr:`all`.

    Attributes:
        _index: The currently bound index, or ``None`` if not yet resolved.
    """

    _index: int | None = eqx.field(static=True, default=None)

    @abstractmethod
    def at_index(self, index: int) -> T:
        """Return the parameter value at a specific dataset index.

        Args:
            index: The dataset index to retrieve.

        Returns:
            The parameter value for the given index.
        """
        pass

    def at(self, index: int) -> T:
        """Alias for :meth:`at_index`."""
        return self.at_index(index)

    def at_current_index(self) -> T:
        if self._index is None:
            raise ValueError(
                "IndexDependentParametrization has not been resolved yet. "
                "Please use the 'resolve_index_dependent_parametrizations' function to resolve all parametrizations "
                "before calling the module inside the jit-boundary. i.e. `resolved_module "
                "resolve_index_dependent_parametrizations(module, index); resolved_module(...).`"
                "When calling outside the jit-boundary, you can use the 'at_index' method to get the parametrization "
                "at a specific index, or use the 'all' property to get all parametrizations."
            )
        return self.at_index(self._index)

    @property
    @abstractmethod
    def n(self) -> int:
        """The number of distinct index values this parameter supports."""
        pass

    @property
    @abstractmethod
    def all(self) -> T:
        """Return all parameter values across all indices."""
        pass


class IndexSliceParameter(IndexDependentParameter[T]):
    """Index-dependent parameter that selects a slice along a given dimension.

    Stores a pytree of arrays where one dimension corresponds to the dataset
    index. Calling :meth:`at_index` slices along that dimension to extract
    the parameter for a single index.

    Attributes:
        parameters: The full pytree of parameters (with an index dimension).
        slice_dim: Static pytree matching ``parameters`` indicating which
            dimension to slice for each leaf.

    Example:
        >>> positions = jnp.zeros((100, 2))  # 100 scan positions, 2D
        >>> param = IndexSliceParameter(positions, dim=0)
        >>> param.at_index(5)  # shape (2,)
    """

    parameters: PyTree
    # slice_dim has same treedef as parameters
    slice_dim: PyTree = eqx.field(static=True)

    def __init__(self, parameters: PyTree, dim: int = 0) -> None:
        """Initialize the index-slice parameter.

        Args:
            parameters: A pytree (array or module) with one dimension
                representing the dataset index.
            dim: The dimension along which to slice. Interpreted relative to
                the start of the array shape (converted to a negative index
                internally for robustness to leading-dimension changes).

        Raises:
            ValueError: If ``parameters`` is a module that does not support
                indexing.
        """

        def get_slice_dim(leaf: Shaped[Array, "..."]) -> int:
            return -len(leaf.shape) + dim if isinstance(leaf, Array) else None

        # Set slice dim to first dimension at initialization time
        # Due to other model transformations, self.parameters shape may change at runtime
        # So long as these transformations only adjust leading dimensions, this is safe.
        resolved_parameters = resolve_array_parametrizations(parameters)
        self.slice_dim = jax.tree.map(get_slice_dim, resolved_parameters)
        if isinstance(parameters, eqx.Module):
            if not hasattr(resolved_parameters, "__getitem__"):
                raise ValueError(
                    f"""The provided module {type(parameters)} does not support indexing.
                    All IndexSliceParameters must support indexing. """
                )
            self.parameters = parameters
            return

        self.parameters = jnp.array(parameters)

    def at_index(self, index: int) -> Shaped[Array, "..."]:
        """Slice the parameters at the given dataset index.

        Args:
            index: Dataset index to select.

        Returns:
            The parameter pytree with the index dimension removed.
        """

        def slice_at_dim(leaf: Shaped[Array, "..."], dim: int) -> Shaped[Array, "..."]:
            if dim is None:
                return leaf
            full_slice = [slice(None)] * len(leaf.shape)
            full_slice[dim] = index
            return leaf[tuple(full_slice)]

        sliced_parameters = jax.tree.map(slice_at_dim, self.parameters, self.slice_dim)
        return sliced_parameters

    @property
    def all(self) -> Shaped[Array, "indices ..."]:
        """The full parameter pytree including the index dimension."""
        return self.parameters

    @property
    def n(self) -> int:
        """The number of indices (size of the slicing dimension)."""
        return resolve_array_parametrizations(self).parameters.shape[0]

    def __getitem__(self, slice: slice) -> Any:  # noqa: ANN401
        raise ValueError(
            "Using __getitem__ on IndexSliceParametrization. "
            "It is likely that the module you are trying to access has not resolved parametrizations yet. "
            "Please use the 'resolve_parametrizations' function to resolve all parametrizations before "
            "calling the module, "
            "i.e. 'resolved_module = resolve_parametrizations(module, index); resolved_module(...)'. "
            "Alternatively, when operating outside the jit-boundary, you can access all parametrizations "
            "via the 'all' property instead when operating"
            "outside the jit-boundary.",
            ValueError,
        )

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        try:
            return super().__getattribute__(name)
        except (AttributeError, TypeError) as e:
            raise AttributeError(
                f"""'IndexSliceParametrization' object has no attribute '{name}'. 
                It is likely that the module you are trying to access has not resolved parametrizations yet. 
                Please use the 'resolve_parametrizations' function to resolve all parametrizations before
                calling the module, 
                i.e. 'resolved_module = resolve_parametrizations(module, index); resolved_module(...)'.
                Alternatively, when operating outside the jit-boundary, you can access all parametrizations 
                via the 'all' property, i.e. 'all_parametrizations = module.foo.all'.
                """
            ) from e


class BoundSliceParameter(eqx.Module):
    """A resolved index-dependent parameter bound to a single index.

    Created by :py:func:`~ptyrax.parametrizations.resolve_index_dependent_parameters`
    when an :py:class:`~ptyrax.parametrizations.IndexDependentParameter` is resolved
    at a specific index. Use :meth:`at_current_index` to access the bound value.

    Attributes:
        parameter: The resolved parameter value for the bound index.
        n: The total number of indices the original parameter supported.
    """

    parameter: PyTree
    n: int = eqx.field(static=True)

    def __init__(self, parameter: PyTree, n: int) -> None:
        """Initialize the bound slice parameter.

        Args:
            parameter: The resolved parameter value for a single index.
            n: Total number of indices in the original parameter.
        """
        self.parameter = parameter
        self.n = n

    def at_index(self, *args) -> T:
        raise ValueError(
            "Using at_index on BoundSliceParameter. ",
            "Likely cause is attempting to use `at_index` while the model IndexableParameter has already been "
            "resolved. If this is the case, simply use the `at_current_index` method instead.",
        )

    def at_current_index(self) -> T:
        """Return the parameter value for the bound index."""
        return self.parameter

    @property
    def all(self) -> T:
        raise ValueError(
            "BoundSliceParameter does not support accessing all parameters. "
            "The model has already been resolved, it is bound to a single index. "
            "Note: write your model without and index dimension, using the 'at_current_index' method to access "
            "any parameter which is a function of the dataset index."
        )


def resolve_index_dependent_parameters(module: T, index: int) -> T:
    """Recursively resolve all index-dependent parameters at a specific index.

    Traverses the pytree and replaces each
    :py:class:`~ptyrax.parametrizations.IndexDependentParameter` with a
    :py:class:`~ptyrax.parametrizations.BoundSliceParameter` containing the
    value at the specified index.

    Args:
        module: The module (or pytree) containing index-dependent parameters.
        index: The dataset index to resolve at.

    Returns:
        A copy of the module with all index-dependent parameters bound to
        the given index.
    """

    def resolve_fn(x: Any) -> Any:  # noqa: ANN401
        if isinstance(x, IndexDependentParameter):
            x_resolved = resolve_array_parametrizations(x)
            x_resolved_index = x_resolved.at(index)
            # Recursion ensures nested parametrizations are also resolved
            return BoundSliceParameter(x_resolved_index, n=x_resolved.n)
        return x

    return jax.tree.map(resolve_fn, module, is_leaf=lambda x: isinstance(x, IndexDependentParameter))


def resolve_index_dependent_parameters_all(module: T) -> T:
    """Recursively resolve all index-dependent parameters, returning all
    indices.

    Traverses the pytree and replaces each
    :py:class:`~ptyrax.parametrizations.IndexDependentParameter` with its
    :attr:`all` property, which includes all index values stacked along the
    leading dimension.

    Args:
        module: The module (or pytree) containing index-dependent parameters.

    Returns:
        A copy of the module with all index-dependent parameters replaced by
        their full (all-index) values.
    """

    def resolve_fn(x: Any) -> Any:  # noqa: ANN401
        if isinstance(x, IndexDependentParameter):
            x_resolved = resolve_array_parametrizations(x)
            x_resolved_all = x_resolved.all
            # Recursion ensures nested parametrizations are also resolved
            return x_resolved_all
        return x

    return jax.tree.map(resolve_fn, module, is_leaf=lambda x: isinstance(x, IndexDependentParameter))
