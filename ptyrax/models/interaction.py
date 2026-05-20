from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Type, Union

import equinox as eqx
import gin
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from jax.scipy.spatial.transform import Rotation as JaxRotation
from jaxtyping import Array, Complex, Float, Shaped
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from tensorboardX import SummaryWriter

from ptyrax.field import CoherentField
from ptyrax.initializers import uniform
from ptyrax.logger import log_image
from ptyrax.models.illumination import NamedLoss
from ptyrax.parametrizations import (
    ArrayParametrization,
    DirectArrayParametrization,
    IndexSliceParameter,
    resolve_parametrizations,
)
from ptyrax.spatial import CoordinateSystem, SamplingGrid, reflect
from ptyrax.utils import phase_only_exp, plot, tree_slice_first, unstack_tree

X_AXIS = jnp.array([1.0, 0.0, 0.0])
Y_AXIS = jnp.array([0.0, 1.0, 0.0])
Z_AXIS = jnp.array([0.0, 0.0, 1.0])


class InteractionModel(eqx.Module):
    """Abstract base class for interaction models in ptychographic
    reconstruction.

    An interaction model describes how an incoming coherent illumination field
    interacts with a sample to produce an outgoing field. Subclasses implement
    specific physical interaction mechanisms such as thin-object Fresnel reflection
    or multi-slice propagation.

    All interaction models are JAX-compatible Equinox modules that can be
    differentiated through and used in JIT-compiled pipelines.

    Attributes:
        coordinates: The coordinate system defining the sample position and
            orientation in the global reference frame.
    """

    coordinates: CoordinateSystem

    @abstractmethod
    def __init__(
        self,
        coordinates: CoordinateSystem,
        sampling: SamplingGrid,
        initializer: Callable[[tuple[int, int]], Complex[Array, "m n"]] = None,
    ) -> None:
        pass

    @abstractmethod
    def __call__(self, input_field: CoherentField) -> CoherentField:
        """Compute the interaction of an illumination field with the sample.

        Args:
            input_field: The incoming coherent illumination field incident on
                the sample.

        Returns:
            The outgoing coherent field after interaction with the sample,
            including updated propagation direction and coordinate system.
        """
        pass

    def __log_epoch__(
        self,
        writer: SummaryWriter,
        epoch: int,
        prefix: str = "",
        **kwargs,
    ) -> None:
        """Log interaction model state to TensorBoard at the end of an epoch.

        Logs sample positions, mean rotation Euler angles, and distance to the
        first scan position.

        Args:
            writer: TensorBoard summary writer instance.
            epoch: Current training epoch number.
            prefix: String prefix for all logged scalar/figure names.
            **kwargs: Additional keyword arguments passed by the training loop.
        """
        translation_internal = resolve_parametrizations(self.coordinates.all.translation_internal)
        fig, ax = plt.subplots(1, 1)
        ax.plot(*translation_internal[:, :2].T, ".")
        writer.add_figure("1_sample/interaction_positions", fig, epoch)
        r = np.array(self.coordinates.all.rotation.as_matrix())
        r = np.mean(r, axis=0)
        euler_angles = JaxRotation.from_matrix(r).as_euler("xyz", degrees=True)
        writer.add_scalar("1_sample/rotation/x", euler_angles[0], epoch)
        writer.add_scalar("1_sample/rotation/y", euler_angles[1], epoch)
        writer.add_scalar("1_sample/rotation/z", euler_angles[2], epoch)
        writer.add_scalar(
            "1_sample/position/distance",
            np.linalg.norm(translation_internal[0]),
            epoch,
        )

    @abstractmethod
    def __regularize__(self) -> list[tuple[str, float]]:
        """Compute regularization losses for this interaction model.

        Returns:
            A tuple of (total_loss, named_losses) where total_loss is the
            scalar sum of all regularization terms, and named_losses is a
            list of :py:class:`~ptyrax.models.illumination.NamedLoss` entries
            identifying each contribution.
        """
        pass


@gin.configurable()
class FresnelReflection(InteractionModel):
    r"""Thin-object Fresnel reflection interaction model.

    Models the interaction of the illumination beam with a thin planar sample
    using a multiplicative approximation. The exit field is computed as:

    $$
    \psi_{\text{exit}}(\mathbf{r}) = O(\mathbf{r}) \cdot P(\mathbf{r})
    $$

    where $O$ is the complex reflection coefficient of the sample and $P$ is
    the probe illumination. The reflected field propagation direction is
    computed by reflecting the incident direction about the sample surface
    normal.

    Attributes:
        coordinates: Sample coordinate system with position and rotation for
            each scan point, wrapped as an :py:class:`~ptyrax.parametrizations.IndexSliceParameter`.
        surface_normal: Unit vector normal to the sample surface in the
            sample's local coordinate frame.
        reflection_coefficient: Complex-valued 2D array representing the
            sample's spatially-resolved reflection coefficient.
        sampling: Pixel grid defining the sample discretization.
        forward_sampling: Optional separate grid for the forward model
            computation. If ``None``, uses the illumination field's sampling.
        regularization_functions: Tuple of callables that compute scalar
            regularization penalties from the reflection coefficient.
        normalize: Whether to apply normalization by the square root of the
            number of pixels.
    """

    coordinates: IndexSliceParameter[CoordinateSystem]
    surface_normal: Float[Array, "3"]
    reflection_coefficient: Shaped[Array, "* m n"]
    sampling: SamplingGrid
    forward_sampling: SamplingGrid
    regularization_functions: tuple[Callable[[Complex[Array, "* m n"]], float]] = eqx.field(static=True)
    normalize: bool = eqx.field(static=True, default=True)

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial shape (m, n) of the sample reflection coefficient array."""
        return self.sampling.shape

    def __init__(
        self,
        coordinates: IndexSliceParameter[CoordinateSystem],
        sampling: SamplingGrid,
        forward_sampling: SamplingGrid = None,
        initializer: Callable[[SamplingGrid], Complex[Array, "m n"]] = uniform,
        parametrization_type: Type[ArrayParametrization] = None,
        regularization_functions: tuple[Callable[[Complex[Array, "* m n"]], float]] = (),
        normalize: bool = True,
    ) -> None:
        """Initialize a thin-object Fresnel reflection model.

        Args:
            coordinates: Coordinate system for the sample, defining translation
                and rotation at each scan position.
            sampling: Sampling grid defining the pixel layout of the sample.
            forward_sampling: Optional separate sampling grid for the forward
                model. If ``None``, the illumination field's sampling is used.
            initializer: Callable that takes a
                :py:class:`~ptyrax.spatial.SamplingGrid` and returns an initial
                complex reflection coefficient array.
            parametrization_type: Optional array parametrization class to wrap
                the reflection coefficient (e.g., for enforcing constraints).
            regularization_functions: Tuple of functions computing scalar
                regularization penalties from the reflection coefficient.
            normalize: If ``True``, multiply the interpolated object by
                ``sqrt(m * n)`` to maintain energy normalization.
        """
        if not isinstance(coordinates, IndexSliceParameter):
            self.coordinates = IndexSliceParameter(coordinates)
        else:
            self.coordinates = coordinates

        self.surface_normal = jnp.array([0.0, 0.0, 1.0])
        self.sampling = sampling
        self.forward_sampling = forward_sampling
        self.reflection_coefficient = (
            parametrization_type(initializer(self.sampling))
            if parametrization_type is not None
            else initializer(self.sampling)
        )
        self.regularization_functions = regularization_functions
        self.normalize = normalize

    def __call__(self, illumination_field: CoherentField, normalize: bool = True) -> CoherentField:
        """Compute the thin-object multiplicative interaction.

        Interpolates the sample reflection coefficient onto the illumination
        grid, multiplies it element-wise with the probe, and constructs the
        reflected output field with updated propagation direction.

        Args:
            illumination_field: Incoming coherent illumination field.
            normalize: Whether to apply pixel-count normalization to the
                interpolated object. If the original array is normalized, this ensures the output remains approximally
                normalized after interpolation.

        Returns:
            The reflected coherent field after sample interaction.
        """
        from ptyrax.spatial import interpolate_grid_to_grid

        sample_coordinates = self.coordinates.at_current_index()
        reflection_coefficient = self.reflection_coefficient
        if self.forward_sampling is None:
            forward_sampling = illumination_field.sampling
        else:
            forward_sampling = self.forward_sampling

        shifted_object = interpolate_grid_to_grid(
            reflection_coefficient,
            self.sampling,
            sample_coordinates,
            (self.forward_sampling if self.forward_sampling is not None else illumination_field.sampling),
            illumination_field.coordinate_system,
            equal_pixel_size=self.forward_sampling is None,
        )
        probe_data = illumination_field()[..., 0]
        if self.forward_sampling is not None:
            probe_data = interpolate_grid_to_grid(
                probe_data,
                illumination_field.sampling,
                illumination_field.coordinate_system,
                self.forward_sampling,
                illumination_field.coordinate_system,
            )
        m0, n0 = self.sampling.shape[:2]
        m1, n1 = illumination_field.shape[:2]
        if normalize:
            normalization_factor = jnp.sqrt(m0 * n0)
            shifted_object = shifted_object * normalization_factor

        new_data = probe_data * shifted_object
        surface_normal = self.surface_normal / jnp.linalg.norm(self.surface_normal)
        sample_z_axis = sample_coordinates.rotation.as_matrix().T @ surface_normal
        new_propagation_direction = reflect(illumination_field.propagation_direction, sample_z_axis)
        new_coordinate_system = CoordinateSystem(
            rotation=sample_coordinates.rotation,
            translation=illumination_field.coordinate_system.translation,
        )
        output_field = CoherentField(
            new_data[..., jnp.newaxis],
            illumination_field.wavelength,
            forward_sampling,
            new_coordinate_system,
            new_propagation_direction,
            illumination_field.spatial_dims,
            illumination_field.vector_dim,
        )
        return output_field

    def __regularize__(self) -> tuple[float, list[NamedLoss]]:
        """Compute regularization losses for the reflection coefficient.

        Evaluates each registered regularization function on the current
        reflection coefficient and accumulates the results.

        Returns:
            A tuple of (total_loss, named_losses) where total_loss is the
            scalar sum of all regularization terms, and named_losses is a
            list of :py:class:`~ptyrax.models.illumination.NamedLoss` entries.
        """
        regularizations = []
        total = 0.0
        for fn in self.regularization_functions:
            value = fn(self.reflection_coefficient())
            total += value
            regularizations.append(NamedLoss(f"reflection_coefficient.{fn.__name__}", value))
        return total, regularizations

    def __plot__(self, *args, **kwargs) -> tuple[Figure, SubplotSpec, Union[Array, list[Array]]]:
        """Plot the reflection coefficient as a complex-valued image.

        Args:
            *args: Positional arguments forwarded to
                :py:func:`~ptyrax.utils.plot`.
            **kwargs: Keyword arguments forwarded to
                :py:func:`~ptyrax.utils.plot`.

        Returns:
            A tuple of (figure, subplot_spec, plotted_array) as produced by
            the plotting utility.
        """
        return plot(self.reflection_coefficient, *args, extent=self.sampling.extent, **kwargs)

    def __log_epoch__(
        self,
        writer: SummaryWriter,
        epoch: int,
        prefix: str = "",
        **kwargs,
    ) -> None:
        """Log reflection coefficient images and diagnostics to TensorBoard.

        Logs a real-space image (with gamma correction), the FFT magnitude
        (log-scaled), the coefficient norm, and surface normal components.

        Args:
            writer: TensorBoard summary writer instance.
            epoch: Current training epoch number.
            prefix: String prefix for all logged entries.
            **kwargs: Additional keyword arguments passed by the training loop.
        """
        super(FresnelReflection, self).__log_epoch__(writer, epoch, prefix=prefix)
        coefficient = self.reflection_coefficient
        coefficient = jax.image.resize(coefficient, (512, 512), method="bilinear")
        log_image(
            writer,
            f"0/1_sample/{prefix}image",
            coefficient,
            epoch,
            gamma=0.5,
            title="sample",
            extent=self.sampling.extent,
        )

        log_image(
            writer,
            f"0/1_sample/{prefix}fft_image",
            jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(coefficient))),
            epoch,
            log10=True,
            title="sample fft",
            extent=self.sampling.to_far_field(1.0, 1.0).extent,
        )

        writer.add_scalar(f"1_sample/{prefix}coefficient_norm", jnp.linalg.norm(coefficient), epoch)
        for axis, name in zip(range(3), ["x", "y", "z"]):
            writer.add_scalar(
                f"1_sample/{prefix}surface_normal/surface_normal_{name}",
                self.surface_normal[axis],
                epoch,
            )


@gin.configurable()
class MultiSlice(InteractionModel):
    r"""Multi-slice interaction model for thick samples.

    Divides the sample into multiple axial slices, each modeled as an
    independent :py:class:`~ptyrax.models.interaction.FresnelReflection`
    interaction. The illumination is near-field propagated to each slice depth,
    interacted, and propagated back. The total exit field is the coherent
    sum of contributions from all slices:

    .. math::
        \psi_{\text{exit}} = \sum_{j=1}^{N}
        \mathcal{P}(z_j)\left[ O_j \cdot \mathcal{P}(z_j)[P] \right]

    where $\mathcal{P}(z)$ denotes near-field propagation by distance $z$
    and $O_j$ is the reflection coefficient of the $j$-th slice.

    Attributes:
        coordinates: Sample coordinate system shared across all slices.
        inner_interactions: Vmapped stack of per-slice interaction models.
        slice_distances: Array of axial distances from the reference plane
            to each slice.
        separable_in_z: If ``True``, uses a single shared interaction model
            for all slices (only varying the propagation distance).
        inverted_bottom: If ``True``, inverts the ordering for the bottom
            half of the slices.
    """

    coordinates: IndexSliceParameter[CoordinateSystem]
    inner_interactions: FresnelReflection
    slice_distances: Float[Array, " n_slices"]
    # TODO remove this parameter, it is a bit of a hack for a specific use case and we can refactor the forward logic
    # to be more composable rather than having this separate path.
    separable_in_z: bool = eqx.field(static=True, default=False)
    inverted_bottom: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        coordinates: CoordinateSystem,
        sampling: SamplingGrid,
        slice_distances: Float[Array, " n_slices"],
        initializer: Callable[[tuple[int, int]], Complex[Array, "m n"]] = uniform,
        parametrization_type: Type[ArrayParametrization] = DirectArrayParametrization,
        regularization_functions: tuple[Callable[[Complex[Array, "* m n"]], float]] = (),
        inner_interactions: tuple[InteractionModel] = None,
        **kwargs,
    ) -> None:
        """Initialize a multi-slice interaction model.

        Either provide pre-built ``inner_interactions`` or specify initialization
        parameters to construct identical slices.

        Args:
            coordinates: Coordinate system for the sample.
            sampling: Sampling grid for the slice reflection coefficients.
            slice_distances: 1D array of axial distances (in meters) from the
                reference plane to each slice.
            initializer: Callable producing an initial reflection coefficient.
            parametrization_type: Parametrization class for the reflection
                coefficients.
            regularization_functions: Regularization functions applied to each
                slice.
            inner_interactions: Optional tuple of pre-built interaction models,
                one per slice. If provided, other initialization parameters
                are ignored for the coefficient arrays.
            **kwargs: Additional keyword arguments. Supports ``separable_in_z``
                and ``inverted_bottom`` overrides.
        """
        if inner_interactions is not None:
            self.inner_interactions = jax.tree.map(
                lambda *v: jnp.stack(v, axis=0),
                *inner_interactions,
            )
        else:

            def make_single_interaction(index: int) -> FresnelReflection:
                return FresnelReflection(
                    coordinates=coordinates,
                    sampling=sampling,
                    initializer=initializer,
                    parametrization_type=parametrization_type,
                    regularization_functions=regularization_functions,
                )

            self.inner_interactions = vmap(make_single_interaction)(jnp.arange(len(slice_distances)))

        self.slice_distances = jnp.asarray(slice_distances)
        self.coordinates = coordinates
        self.separable_in_z = kwargs.get("separable_in_z", self.separable_in_z)
        self.inverted_bottom = kwargs.get("inverted_bottom", self.inverted_bottom)

    @classmethod
    def from_interactions(
        cls, interactions: tuple[InteractionModel], distances: Float[Array, " n_slices"], **kwargs
    ) -> Type["MultiSlice"]:
        """Construct a MultiSlice from pre-built per-slice interaction models.

        Args:
            interactions: Tuple of interaction models, one per slice.
            distances: Array of axial slice distances.
            **kwargs: Additional keyword arguments forwarded to the constructor
                (e.g., ``separable_in_z``, ``inverted_bottom``).

        Returns:
            A new :py:class:`~ptyrax.models.interaction.MultiSlice` instance
            wrapping the provided interactions.
        """
        return cls(
            coordinates=interactions[0].coordinates,
            sampling=interactions[0].sampling,
            slice_distances=distances,
            inner_interactions=interactions,
            **kwargs,
        )

    @property
    def n_slices(self) -> int:
        """Number of axial slices in the model."""
        return len(self.slice_distances)

    def __regularize__(self) -> tuple[float, list[NamedLoss]]:
        """Compute regularization losses aggregated over all slices.

        Returns:
            A tuple of (total_loss, named_losses) delegated to the inner
            vmapped interaction stack.
        """
        return self.inner_interactions.__regularize__()

    # TODO this method is a bit of a hack for a specific use case. We may want to refactor the forward logic to be more
    # composable rather than having this separate path.
    def separable_forward_fields(self, illumination_field: CoherentField) -> CoherentField:
        """Compute per-slice reflected fields using a single shared
        interaction.

        Uses only the first slice's interaction model for all depths,
        varying only the propagation distance. This is the forward pass
        when ``separable_in_z=True``.

        Args:
            illumination_field: Incoming coherent illumination field.

        Returns:
            A vmapped stack of reflected fields, one per slice.
        """
        top_interaction = tree_slice_first(self.inner_interactions)

        backreflected_fields = eqx.filter_vmap(self.single_slice_forward, in_axes=(None, 0, None))(
            illumination_field,
            self.slice_distances,
            top_interaction,
        )
        return backreflected_fields

    def forward_fields(self, illumination_field: CoherentField) -> CoherentField:
        """Compute per-slice reflected fields using independent interactions.

        Each slice uses its own interaction model. This is the forward pass
        when ``separable_in_z=False``.

        Args:
            illumination_field: Incoming coherent illumination field.

        Returns:
            A vmapped stack of reflected fields, one per slice.
        """
        backreflected_fields = eqx.filter_vmap(self.single_slice_forward, in_axes=(None, 0, 0))(
            illumination_field,
            self.slice_distances,
            self.inner_interactions,
        )
        return backreflected_fields

    @staticmethod
    def single_slice_forward(
        illumination_field: CoherentField,
        z_displacement: float,
        interaction: InteractionModel,
    ) -> CoherentField:
        """Forward model for a single slice: propagate, interact, propagate back.

        Args:
            illumination_field: Incoming coherent illumination field.
            z_displacement: Axial distance from the reference plane to this
                slice (in meters).
            interaction: The interaction model for this slice.

        Returns:
            The back-reflected field after round-trip propagation and
            interaction with the slice.
        """
        propagated_probe = illumination_field.propagate_tilted_nearfield(z_displacement * Z_AXIS)
        post_interaction_field = interaction(propagated_probe, normalize=True)
        backreflected_field = post_interaction_field.propagate_tilted_nearfield(z_displacement * Z_AXIS)

        return backreflected_field

    def __call__(self, illumination_field: CoherentField, normalize: bool = True) -> CoherentField:
        """Compute the multi-slice interaction as a coherent sum over slices.

        Propagates the illumination to each slice depth, computes the
        per-slice interaction, propagates back, and sums all contributions
        coherently.

        Args:
            illumination_field: Incoming coherent illumination field.
            normalize: Whether to apply normalization in the per-slice
                interactions.

        Returns:
            The total reflected coherent field (coherent sum of all slices).
        """
        backreflected_fields = (
            self.separable_forward_fields(illumination_field)
            if self.separable_in_z
            else self.forward_fields(illumination_field)
        )
        total_field_data = jnp.sum(backreflected_fields(), axis=0)
        single_field = jax.tree.map(lambda leaf: leaf[0], backreflected_fields)
        total_field = eqx.tree_at(lambda f: f.data, single_field, total_field_data)
        return total_field

    def __log_epoch__(self, writer: SummaryWriter, epoch: int, prefix: str = "", **kwargs) -> None:
        """Log per-slice distances and inner interaction diagnostics.

        Args:
            writer: TensorBoard summary writer instance.
            epoch: Current training epoch number.
            prefix: String prefix for all logged entries.
            **kwargs: Additional keyword arguments forwarded to inner
                interaction logging.
        """
        for i in range(self.n_slices):
            writer.add_scalar(f"1_sample/{prefix}slice_distance_{i}", self.slice_distances[i], epoch)

        writer.add_scalar(
            f"1_sample/{prefix}_distance12_abs", jnp.abs(self.slice_distances[1] - self.slice_distances[0]), epoch
        )

        for i in range(self.n_slices):
            jax.tree.map(
                lambda leaf: leaf[i],
                self.inner_interactions,
            ).__log_epoch__(writer, epoch, f"{prefix}inner_{i}_", **kwargs)


@gin.configurable
def to_single_slice_approximation(
    multislice: MultiSlice,
    tilt_angle: float = 0.0,
    wavelength: float = 1.0,
    interaction_type: type[InteractionModel] = FresnelReflection,
) -> FresnelReflection:
    r"""Convert a multi-slice model to a single-slice approximation.

    Collapses the multi-slice reflection coefficients into a single effective
    reflection coefficient by coherently summing the per-slice contributions
    with depth-dependent phase factors:

    .. math::
        O_{\text{eff}}(\mathbf{r}) = \sum_j O_j(\mathbf{r})
        \exp\!\left(i \frac{4\pi}{\lambda} z_j \cos\theta\right)

    where $z_j$ is the slice distance and $\theta$ is the tilt angle.

    Args:
        multislice: The multi-slice interaction model to collapse.
        tilt_angle: Beam tilt angle in degrees relative to the surface
            normal. Affects the depth-dependent phase factor.
        wavelength: Illumination wavelength (in meters).
        interaction_type: Class of the output single-slice interaction model.

    Returns:
        A single-slice interaction model with the coherently summed
        reflection coefficient.

    Raises:
        TypeError: If ``multislice`` is not a
            :py:class:`~ptyrax.models.interaction.MultiSlice` instance.
    """
    if not isinstance(multislice, MultiSlice):
        raise TypeError(f"Expected MultiSlice, got {type(multislice)}")
    interaction_data = multislice.inner_interactions.reflection_coefficient()
    phases = 2 * np.pi / wavelength * multislice.slice_distances * 2 * jnp.cos(jnp.deg2rad(tilt_angle))
    phases = phases.reshape((-1, 1, 1))
    new_interaction_data = jnp.sum(interaction_data * phase_only_exp(phases), axis=0)
    first_coordinates = unstack_tree(multislice.inner_interactions.coordinates)[0]
    first_sampling = unstack_tree(multislice.inner_interactions.sampling)[0]
    single_interaction = interaction_type(
        coordinates=first_coordinates,
        sampling=first_sampling,
        initializer=lambda shape: new_interaction_data,
    )
    return single_interaction
