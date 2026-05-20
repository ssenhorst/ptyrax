from collections import namedtuple
from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax.numpy as jnp
import numpy as np
import periodictable.xsf as xsf
from jaxtyping import Array, Float

ANGSTROM = 1e-10  # meters


def fresnel_coefficients(n1: float, n2: float, theta_i_deg: float, wavelength: float) -> NamedTuple:
    """Compute Fresnel reflection amplitudes and reflectances for an interface
    between medium 1 (incident) and medium 2 (transmitted).

    Handles complex refractive indices and ensures physical energy.
    """
    theta_i = np.deg2rad(theta_i_deg)

    # Compute transmitted angle using Snell's law (complex arcsin)
    sin_theta_t = n1 / n2 * np.sin(theta_i)
    theta_t = np.arcsin(sin_theta_t)

    # Fresnel amplitudes
    cos_theta_i = np.cos(theta_i)
    cos_theta_t = np.cos(theta_t)

    r_s = (n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)
    r_p = (n2 * cos_theta_i - n1 * cos_theta_t) / (n2 * cos_theta_i + n1 * cos_theta_t)

    t_s = 2 * n1 * cos_theta_i / (n1 * cos_theta_i + n2 * cos_theta_t)
    t_p = 2 * n1 * cos_theta_i / (n2 * cos_theta_i + n1 * cos_theta_t)

    # Reflectances (physical) using real part of Poynting flux
    R_s = abs(r_s) ** 2  # noqa: N806
    R_p = abs(r_p) ** 2  # noqa: N806

    # Transmittance (energy fraction), includes absorption in medium 2
    T_s = (n2.real * cos_theta_t.real) / (n1.real * cos_theta_i.real) * abs(t_s) ** 2  # noqa: N806
    T_p = (n2.real * cos_theta_t.real) / (n1.real * cos_theta_i.real) * abs(t_p) ** 2  # noqa: N806

    # Energy check
    energy_check_s = R_s + T_s
    energy_check_p = R_p + T_p

    Coefficients = namedtuple(
        "FresnelCoefficients",
        ["r_s", "r_p", "R_s", "R_p", "t_s", "t_p", "T_s", "T_p", "energy_check_s", "energy_check_p"],
    )
    return Coefficients(
        r_s=r_s,
        r_p=r_p,
        R_s=R_s,
        R_p=R_p,
        t_s=t_s,
        t_p=t_p,
        T_s=T_s,
        T_p=T_p,
        energy_check_s=energy_check_s,
        energy_check_p=energy_check_p,
    )


def fresnel_coefficients_fourier(
    n1: complex,
    n2: complex,
    k_z_1: Float[Array, " m n"],
    k_z_2: Float[Array, " m n"],
    pol: Literal["s", "p"] = "p",
) -> NamedTuple:
    """Compute Fresnel reflection amplitudes and reflectances for an interface
    between medium 1 (incident) and medium 2 (transmitted).

    Handles complex refractive indices and ensures physical energy.
    """
    if pol == "s":
        denominator = k_z_1 + k_z_2
        r_s = (k_z_1 - k_z_2) / denominator
        t_s = 2 * k_z_1 / denominator
        return r_s, t_s
    elif pol == "p":
        denominator = n2**2 * k_z_1 + n1**2 * k_z_2
        r_p = (n2**2 * k_z_1 - n1**2 * k_z_2) / denominator
        t_p = 2 * n1 * n2 * k_z_1 / denominator
        return r_p, t_p
    else:
        raise ValueError(f"Invalid polarization: {pol!r}. Must be 's' or 'p'.")


def transmission_coefficient(
    from_material: str,
    to_material: str,
    wavelength: float,  # meters
    angle_of_incidence: float,
    polarization: Literal["s", "p"] = "p",
) -> complex:
    """Compute the Fresnel transmission amplitude at an interface.

    Args:
        from_material: Incident medium identifier (or ``"vacuum"``/``"air"``).
        to_material: Transmitted medium identifier.
        wavelength: Photon wavelength in meters.
        angle_of_incidence: Angle of incidence in degrees.
        polarization: Polarization state (``"s"`` or ``"p"``).

    Returns:
        Complex Fresnel transmission amplitude coefficient.
    """
    n1 = (
        index_of_refraction(from_material, wavelength=wavelength)
        if from_material not in [None, "vacuum", "air"]
        else 1.0
    )
    n2 = index_of_refraction(to_material, wavelength=wavelength) if to_material not in [None, "vacuum", "air"] else 1.0
    coeffs = fresnel_coefficients(n1, n2, angle_of_incidence, wavelength)
    if polarization == "s":
        return coeffs.t_s
    elif polarization == "p":
        return coeffs.t_p
    else:
        raise ValueError(f"Invalid polarization: {polarization!r}. Must be 's' or 'p'.")


def index_of_refraction(
    material: str,
    wavelength: float,
    density: float | None = None,
    **kwargs,
) -> complex:
    """Get the complex index of refraction for a given material at a specified
    wavelength.

    Wavelength is in meters. Additional keyword arguments can be passed
    to the underlying function.
    """
    if wavelength > 1e-3:
        raise ValueError(
            f"Wavelength was very large for index of refraction lookup: {wavelength}."
            " Please provide wavelength in meters."
        )
    if material in MATERIALS.keys():
        if MATERIALS[material].refractive_index is not None:
            return MATERIALS[material].refractive_index
        density = MATERIALS[material].density
        compound = MATERIALS[material].chemical_formula
    else:
        compound = material

    return xsf.index_of_refraction(compound, wavelength=wavelength / ANGSTROM, density=density, **kwargs)


@dataclass
class Material:
    """Physical material properties for optical calculations.

    Attributes:
        density: Mass density in g/cm³, or None to use tabulated values.
        chemical_formula: Chemical formula string for lookup (e.g. ``"Si3N4"``).
        refractive_index: Fixed refractive index override, or None for
            wavelength-dependent lookup.
    """

    density: float | None = None  # g/cm^3
    chemical_formula: str | None = None
    refractive_index: float | None = None


MATERIALS = {
    "vacuum": Material(density=0.0, chemical_formula="", refractive_index=1.0),
    "air": Material(density=0.0, chemical_formula="", refractive_index=1.0),
    "PMMA": Material(density=1.18, chemical_formula="C5O2H8"),
    "Si3N4": Material(density=3.17, chemical_formula="Si3N4"),
    "SiO2": Material(density=2.20, chemical_formula="SiO2"),
}


def reflection_coefficient(
    from_material: str,
    to_material: str,
    wavelength: float,
    angle_of_incidence: float,
    polarization: Literal["s", "p"] = "p",
    from_kwargs: dict = None,
    to_kwargs: dict = None,
) -> complex:
    """Compute the Fresnel reflection amplitude at an interface.

    Args:
        from_material: Incident medium identifier.
        to_material: Reflecting medium identifier.
        wavelength: Photon wavelength in meters.
        angle_of_incidence: Angle of incidence in degrees.
        polarization: Polarization state (``"s"`` or ``"p"``).
        from_kwargs: Extra kwargs for incident material index lookup.
        to_kwargs: Extra kwargs for reflecting material index lookup.

    Returns:
        Complex Fresnel reflection amplitude coefficient.
    """
    if from_kwargs is None:
        from_kwargs = {}
    if to_kwargs is None:
        to_kwargs = {}
    n1 = (
        index_of_refraction(from_material, wavelength=wavelength, **from_kwargs)
        if from_material not in [None, "vacuum", "air"]
        else 1.0
    )
    n2 = (
        index_of_refraction(to_material, wavelength=wavelength, **to_kwargs)
        if to_material not in [None, "vacuum", "air"]
        else 1.0
    )
    coeffs = fresnel_coefficients(n1, n2, angle_of_incidence, wavelength)
    if polarization == "s":
        return coeffs.r_s
    elif polarization == "p":
        return coeffs.r_p
    else:
        raise ValueError(f"Invalid polarization: {polarization!r}. Must be 's' or 'p'.")


def absorption_coefficient(
    material: str, wavelength: float, z: float, angle_of_incidence: float, from_material: str = "vacuum"
) -> float:
    r"""Compute the Beer–Lambert absorption factor for propagation through a
    material.

    Calculates $\exp(\alpha z)$ where $\alpha$ is the imaginary part of the
    wave-vector component along the propagation direction.

    Args:
        material: Material identifier for the absorbing medium.
        wavelength: Photon wavelength in meters.
        z: Propagation distance in meters.
        angle_of_incidence: Angle of incidence in degrees.
        from_material: Incident medium (default ``"vacuum"``).

    Returns:
        Real-valued absorption factor (dimensionless).
    """
    if material in {"vacuum", "air", None}:
        return 1.0
    n_1 = index_of_refraction(from_material, wavelength=wavelength)
    n_2 = index_of_refraction(material, wavelength=wavelength)
    k_0 = 2 * np.pi / wavelength  # 1/m
    k_x = k_0 * n_1 * jnp.sin(jnp.rad2deg(angle_of_incidence))
    k_z = jnp.sqrt((k_0 * n_2) ** 2 - k_x**2)
    alpha = k_z.imag
    return jnp.exp(alpha * z)
