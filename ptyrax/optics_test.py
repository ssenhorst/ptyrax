import jax.numpy as jnp
import numpy as np
import pytest

from ptyrax.optics import fresnel_coefficients_fourier, index_of_refraction


def _kz_from_angle(n: complex, wavelength_m: float, theta_deg: float) -> complex:
    k_x = np.sin(np.deg2rad(theta_deg)) / wavelength_m
    return np.sqrt((n / wavelength_m) ** 2 - k_x**2)


def _widget_style_reflectivity_s_vacuum_to_si(wavelength_m: float, theta_deg: float) -> float:
    n0 = 1.0 + 0.0j
    n_si = index_of_refraction("Si", wavelength_m)
    kz0 = _kz_from_angle(n0, wavelength_m, theta_deg)
    kz2 = _kz_from_angle(n_si, wavelength_m, theta_deg)
    r, _ = fresnel_coefficients_fourier(
        n1=n0,
        n2=n_si,
        k_z_1=jnp.array([[kz0]]),
        k_z_2=jnp.array([[kz2]]),
        pol="s",
    )
    return float(np.abs(np.asarray(r)[0, 0]) ** 2)


def test_import_optics() -> None:
    __import__("ptyrax.optics")


def test_fresnel_coefficients_fourier_rejects_invalid_polarization() -> None:
    with pytest.raises(ValueError, match="Invalid polarization"):
        fresnel_coefficients_fourier(
            n1=1.0 + 0.0j,
            n2=1.0 + 0.0j,
            k_z_1=jnp.ones((1, 1)),
            k_z_2=jnp.ones((1, 1)),
            pol="invalid",
        )


def test_fresnel_coefficients_fourier_matches_vacuum_si_reference_table() -> None:
    # Reference values (energy reflectivity) from CXRO / reflection widget setup:
    # vacuum -> Si, s polarization, 70 degrees incidence (20 degrees from grazing).
    reference_table = [
        (13.0000, 1.573746e-04),
        (13.3500, 6.081289e-05),
        (13.7000, 1.455097e-04),
        (14.0500, 3.452014e-04),
        (14.4000, 6.378869e-04),
        (14.7500, 1.031688e-03),
        (15.1000, 1.522389e-03),
        (15.4500, 2.107699e-03),
        (15.8000, 2.819326e-03),
        (16.1500, 3.629815e-03),
        (16.5000, 4.466000e-03),
        (16.8500, 5.344948e-03),
        (17.2000, 6.327143e-03),
        (17.5500, 7.368452e-03),
        (17.9000, 8.448690e-03),
        (18.2500, 9.675002e-03),
        (18.6000, 1.105483e-02),
        (18.9500, 1.256918e-02),
        (19.3000, 1.421488e-02),
        (19.6500, 1.611716e-02),
        (20.0000, 1.827620e-02),
    ]
    theta_deg = 70.0

    for wavelength_nm, reference_reflectivity in reference_table:
        wavelength_m = wavelength_nm * 1e-9
        n0 = 1.0 + 0.0j
        n_si = index_of_refraction("Si", wavelength_m)

        kz0 = _kz_from_angle(n0, wavelength_m, theta_deg)
        kz2 = _kz_from_angle(n_si, wavelength_m, theta_deg)

        r_s, _ = fresnel_coefficients_fourier(
            n1=n0,
            n2=n_si,
            k_z_1=jnp.array([[kz0]]),
            k_z_2=jnp.array([[kz2]]),
            pol="s",
        )
        reflectivity = float(np.abs(np.asarray(r_s)[0, 0]) ** 2)
        widget_reflectivity = _widget_style_reflectivity_s_vacuum_to_si(wavelength_m, theta_deg)

        # Match the hard-coded reference table and cross-check against the widget computation path.
        assert np.isclose(reflectivity, reference_reflectivity, rtol=0.08, atol=2e-4)
        assert np.isclose(reflectivity, widget_reflectivity, rtol=1e-9, atol=1e-12)
