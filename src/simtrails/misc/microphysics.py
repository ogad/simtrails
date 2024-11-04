# %%
import numpy as np

from simtrails.atmosphere import Atmosphere


standard_atmosphere = Atmosphere.from_name("us")


def sph_particle_radius(iwc, n_ice):
    """Get the volume radius, given the ice water content and number concentration of ice particles."""
    return ((3 * iwc) / (4 * np.pi * n_ice)) ** (1 / 3)  # Volume radius assumption


def eff_radius_from_vol(vol_radius, rh_ice):
    """Convert  volume radius to effective radius, using the Schumann2011 conversion."""
    return vol_radius / radius_ratio_sch11(vol_radius, rh_ice)


def vol_radius_from_eff(eff_radius, rh_ice):
    """Convert effective radius to volume radius, using the Schumann2011 conversion."""
    return eff_radius * radius_ratio_sch11(eff_radius, rh_ice)


def iwc_from_path(iwp, depth):
    """Convert IWP to IWC."""
    return iwp / (depth)


def iwc_kgkg1_to_gm3(iwc, altitude):
    """Convert the units of IWP (from per mass or air to per volume of air)."""
    density = standard_atmosphere.value_at_altitude(
        "air", altitude
    )  # molecules cm-3 for air
    density = density * 1e6  # cm-3 to m-3
    density = density * 28.96 / 6.022e23  # m-3 to g/m3
    iwc = iwc  # kg/kg  = g/g
    iwc = iwc * density  # g/g to g/m^3
    return iwc


def radius_ratio_sch11(vol_radius, rh_ice):
    """The ratio of volume to effective radius, as given by Schumann2011."""
    # radii in microns; r_vol / r_eff
    # from schumann2011
    r_0 = 1
    r_1 = 20
    r_r0 = vol_radius / r_0
    r_r1 = vol_radius / r_1

    c_habit = 2.2 + 0.00113 * r_r0 - 1.121 * np.exp(-0.011 * r_r0)
    c_r = 0.9 + (c_habit - 1.7) * (1 - np.exp(-1.0 * r_r1))
    c = 1 + (c_r - 1) * (1 - np.exp(-1.0 * rh_ice)) / (1 - np.exp(-1.0))

    return c


# %%
if __name__ == "__main__":
    print(standard_atmosphere.value_at_altitude("air", 11))
    print(iwc_kgkg1_to_gm3(1e-5, 11))

# %%


def ic_lrt_properties(
    iwc: float = None,
    iwp: float = None,
    eff_radius: float = None,
    n_ice: float = None,
    depth: float = 0.5,
):
    """
    Calculate the properties (IWP and effective radius) of an ice contrail in the lower radiative troposphere.

    Args:
        iwc (float, optional): Peak ice water content in g/m^3. Defaults to None.
        iwp (float, optional): Peak ice water path in g/m^2. Defaults to None.
        eff_radius (float, optional): Effective radius of ice particles in micrometers. Defaults to None.
        n_ice (float, optional): Number concentration of ice particles in L^-1. Defaults to None.
        depth (float, optional): Thickness of the contrail in km. Defaults to 0.5.

    Returns:
        tuple: A tuple containing the peak ice water content in g/m^3 and the effective radius of ice particles in micrometers.
    """
    if iwc is None and iwp is None:
        iwp = 5.0

    if iwp is None:
        iwp = iwc * depth * 1e3
        # iwc = iwc_from_path(iwp, depth * 1e3)

    if eff_radius is None and n_ice is None:
        eff_radius = 10
    elif eff_radius is None:
        eff_radius = sph_particle_radius(iwc, n_ice * 1e3) * 1e6
    return iwp, eff_radius
