from pathlib import Path
import numpy as np

from simtrails.atmosphere import Atmosphere

standard_atmosphere = Atmosphere(
    Path(__file__).parent / "data" / "atmospheres" / "afglus.dat"
)


def get_tau(iwp=None, iwc=None, depth=None, eff_radius=None, n_ice=None, **_):
    """
    Calculate the optical depth (tau) of ice particles in the atmosphere.

    To combine with radiative_transfer.get_optical_depth().

    Parameters:
    iwp (float): Ice water path in g/m^2 (default: None)
    iwc (float): Ice water content in g/m^3 (default: None)
    depth (float): Thickness of ice layer in meters (default: None)
    eff_radius (float): Effective radius of ice particles in micrometers (default: None)
    n_ice (float): Number concentration of ice particles in cm^-3 (default: None)

    Returns:
    float: Optical depth (tau) of ice particles in the atmosphere.
    """
    import logging
    from simtrails.misc.microphysics import sph_particle_radius

    if iwp is None:
        iwp = iwc * depth * 1e3  # g/m^2
    elif iwc is not None and depth is not None:
        logging.warning("Ignoring iwc and depth, using iwp instead")
    if eff_radius is None:
        eff_radius = sph_particle_radius(iwp, n_ice * 1e3) * 1e6  # um
    r = eff_radius * 1e-6  # m
    rho_ice = 0.917e6  # g/m^3
    return 2 / 3 * iwp / (rho_ice * r)


def atm_stf_blz(T):
    """Calculate the blackbody longwave radiation at the given temperature, using an approximation of the Stefan-Boltzmann law."""
    return 1.607e-4 * T**2.528


def lw_cloud(lw_inc, T_c, tau):
    """
    Calculate the longwave radiation flux in the presence of clouds.
    Depricated (not used).  

    Parameters:
    lw_inc (float): Incoming longwave radiation flux.
    T_c (float): Cloud temperature.
    tau (float): Cloud optical depth.

    Returns:
    float: Longwave radiation flux in the presence of clouds.
    """
    cld_upwelling = atm_stf_blz(T_c)
    f_unobscured = np.exp(-tau)
    return f_unobscured * lw_inc + (1 - f_unobscured) * cld_upwelling


def sw_cloud(tau, sw_inc=1361, g=0.85):
    """
    Calculate the shortwave radiation after accounting for cloud albedo.
    Depricated (not used).

    Parameters:
    tau (float): Optical depth of the cloud.
    sw_inc (float, optional): Incoming shortwave radiation. Default is 1361 W/m^2.
    g (float, optional): Cloud asymmetry parameter. Default is 0.85.

    Returns:
    float: Shortwave radiation after accounting for cloud albedo.
    """
    cloud_albedo = (1 - g) * tau / (2 + (1 - g) * tau)
    return sw_inc * (1 - cloud_albedo)


def get_rf(
    iwc,
    depth,
    eff_radius,
    altitude,
    atmosphere=standard_atmosphere,
    solar_zenith_angle=0,
    f_solar=1,
):
    """
    Calculate the radiative forcing (rf) due to cloud ice particles.
    Depricated (not used).


    Parameters:
    iwc (float): Ice water content in kg/m^3.
    depth (float): Thickness of the cloud layer in meters.
    eff_radius (float): Effective radius of the ice particles in meters.
    base (float): Base altitude of the cloud layer in meters.
    atmosphere (Atmosphere, optional): Atmosphere object representing the atmospheric conditions. Defaults to standard_atmosphere.
    solar_zenith_angle (float, optional): Solar zenith angle in radians. Defaults to 0.
    f_solar (float, optional): Solar scaling factor. Defaults to 1.

    Returns:
    float: The radiative forcing (rf) in W/m^2.
    """
    T_s = atmosphere.value_at_altitude("T", 0)
    T_c = atmosphere.value_at_altitude("T", altitude + depth)

    tau = get_tau(iwc, depth, eff_radius)

    solar_constant = 1361 * f_solar * np.cos(np.pi / 2 - solar_zenith_angle)  # W/m^2
    if solar_zenith_angle > np.pi / 2:
        solar_constant = 0
    tau_solar = tau / np.cos(solar_zenith_angle)

    sw_forced_downwelling = sw_cloud(tau_solar, sw_inc=solar_constant)
    lw_forced_upwelling = lw_cloud(atm_stf_blz(T_s), T_c, tau)

    sw_rf = sw_forced_downwelling - solar_constant
    lw_rf = atm_stf_blz(T_s) - lw_forced_upwelling

    rf = lw_rf + sw_rf

    return rf


def get_rf_schumann12(
    iwc,
    depth,
    eff_radius,
    altitude,
    atmosphere=standard_atmosphere,
    solar_zenith_angle=0.0,
    albedo=0.0,
    olr=200.0,
    f_solar=1,
    sw=True,
    lw=True,
):
    """
    Calculate the radiative forcing using the Schumann et al. (2012) method.
    Depricated (not used).

    Args:
        iwc (float or numpy.ndarray): Ice water content.
        depth (float or numpy.ndarray): Cloud thickness.
        eff_radius (float or numpy.ndarray): Effective radius.
        base (float or numpy.ndarray): Cloud base altitude.
        atmosphere (Atmosphere, optional): Atmosphere object representing the atmospheric conditions. Defaults to standard_atmosphere.
        solar_zenith_angle (float or numpy.ndarray, optional): Solar zenith angle in degrees. Defaults to 0.0.
        albedo (float or numpy.ndarray, optional): Surface albedo. Defaults to 0.0.
        olr (float or numpy.ndarray, optional): Outgoing longwave radiation. Defaults to 200.0.
        f_solar (float or numpy.ndarray, optional): Solar flux scaling factor. Defaults to 1.
        sw (bool, optional): Flag indicating whether to calculate shortwave radiative forcing. Defaults to True.
        lw (bool, optional): Flag indicating whether to calculate longwave radiative forcing. Defaults to True.

    Returns:
        float or numpy.ndarray: Radiative forcing.

    """

    from itertools import zip_longest

    from pycontrails.models.cocip.radiative_forcing import (
        shortwave_radiative_forcing,
        longwave_radiative_forcing,
    )
    from pycontrails.physics.geo import (
        solar_direct_radiation,
        solar_constant,
        orbital_position,
    )

    (
        iwc,
        depth,
        eff_radius,
        altitude,
        solar_zenith_angle,
        f_solar,
        albedo,
        olr,
    ) = np.atleast_1d(
        iwc, depth, eff_radius, altitude, solar_zenith_angle, f_solar, albedo, olr
    )

    max_shape = [
        max(items)
        for items in zip_longest(
            *[
                arr.shape
                for arr in [
                    iwc,
                    depth,
                    eff_radius,
                    altitude,
                    solar_zenith_angle,
                    f_solar,
                    albedo,
                    olr,
                ]
            ],
            fillvalue=0,
        )
    ]
    iwc, depth, eff_radius, altitude, solar_zenith_angle, f_solar, albedo, olr = [
        np.broadcast_to(arg, max_shape)
        for arg in [
            iwc,
            depth,
            eff_radius,
            altitude,
            solar_zenith_angle,
            f_solar,
            albedo,
            olr,
        ]
    ]

    times = np.broadcast_to([np.datetime64("2019-12-20 12:00")], max_shape)
    sd0 = solar_constant(
        orbital_position(times)
    )  # only used for sza calculation:  #datetime...
    sdr = np.cos(solar_zenith_angle) * sd0
    rsr = sdr * albedo  # zero surface albedo, zero background cloud case
    olr = olr  # ???????
    air_temp = np.broadcast_to(
        atmosphere.value_at_altitude("T", altitude + depth), max_shape
    )

    tau = get_tau(iwc, depth, eff_radius)
    habit_weights = np.broadcast_to(
        np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]]), max_shape + [8]
    )

    rf_sw = (
        shortwave_radiative_forcing(
            eff_radius,
            sdr,
            rsr,
            sd0,
            tau,
            np.broadcast_to(0.0, max_shape),
            habit_weights,
            eff_radius,
        )
        if sw
        else np.broadcast_to(0.0, max_shape)
    )
    rf_lw = (
        longwave_radiative_forcing(
            eff_radius,
            olr,
            air_temp,
            tau,
            np.broadcast_to(0.0, max_shape),
            habit_weights,
            eff_radius,
        )
        if lw
        else np.broadcast_to(0.0, max_shape)
    )
    rf = rf_sw + rf_lw
    return rf
