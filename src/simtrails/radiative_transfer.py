import numpy as np
from pyLRT import get_lrt_folder

# from pyLRT.cloud import Cloud
from pathlib import Path
import os

from .atmosphere import Atmosphere

try:
    LIBRADTRAN_FOLDER = Path(os.environ["LIBRADTRAN_FOLDER"])
except KeyError:
    LIBRADTRAN_FOLDER = get_lrt_folder()


LATITUDES = {
    "us": 40.0,
    "t": 0.0,
    "ms": 45.0,
    "mw": 45.0,
    "ss": 60.0,
    "sw": 60.0,
}
LONGITUDES = {
    "us": 0.0,
    "t": 0.0,
    "ms": 0.0,
    "mw": 0.0,
    "ss": 0.0,
    "sw": 0.0,
}
DATES = {
    "us": "06 01",
    "t": "06 01",
    "ms": "06 01",
    "mw": "12 01",
    "ss": "06 01",
    "sw": "12 01",
}
ATMOSPHERE_NAMES = {
    "us": "US-standard",
    "t": "tropics",
    "ms": "midlatitude_summer",
    "mw": "midlatitude_winter",
    "ss": "subarctic_summer",
    "sw": "subarctic_winter",
}
SOLAR_CHANNELS = [f"goes-r_abi_ch{i:02d}" for i in range(7)]


def get_output(cloud=None, brightness=True, **kwargs):
    """
    Run the radiative transfer model and return the output.

    Parameters:
        cloud: Cloud object representing the cloud properties, this is not normally used (optional)
        brightness: Flag indicating whether to compute brightness quantities (default: True)
        kwargs: Additional keyword arguments to customize the radiative transfer model

    Returns:
        rt_output: Output of the radiative transfer model
    """
    from pyLRT import RadTran, OutputParser

    rt = RadTran(LIBRADTRAN_FOLDER)
    rt.options["zout"] = "TOA"
    rt.options["albedo"] = "0"
    rt.options["umu"] = "-1.0 1.0"
    rt.options["output_user"] = "lambda edir eup uu"
    if brightness:
        rt.options["output_quantity"] = "brightness"
    if kwargs.get("mol_abs_param") != "KATO2":
        rt.parser = OutputParser(dims=["lambda"])

    for option in kwargs:
        rt.options[option] = kwargs[option]

    if cloud is not None:
        rt.cloud = cloud

    try:
        rt_output = rt.run(quiet=True)
    except ValueError as e:
        print("rt.options:")
        for k, v in rt.options.items():
            print(f"{k}: {v}")
        print("rt.cloud:")
        print(rt.cloud)
        raise e

    return rt_output


def get_ic_file(iwc, altitude, depth, eff_radius, background="Clear"):
    """
    Generate an ice cloud file and return its path.

    Parameters:
        iwc: Ice water content (g/m^3)
        altitude: Base altitude of the cloud (km)
        depth: Thickness of the cloud (km)
        eff_radius: Effective radius of the cloud particles (microns)

    Returns:
        ic_file: Path to the generated ice cloud file
    """
    import tempfile
    from simtrails.forcing import get_tau

    top = altitude + (depth / 2)  # km
    bottom = altitude - (depth / 2)  # km
    cirrus_top = 8.0  # km
    cirrus_bottom = 7.5  # km
    # cirrus_iwc = 0.05  # g/m^3 - 5um cirruc
    # cirrus_eff_radius = 5.0  # microns - parameterisation overlap
    cirrus_iwc = 0.25  # g/m^3 - 30um cirrus
    cirrus_eff_radius = (
        30.0  # microns - calculated specially for mie tables (revisions)
    )

    cloudstr = [
        f"{top:.4f} 0 0",
        f"{bottom:.4f} {iwc:.4f} {eff_radius:.4f}",
    ]

    if background == "8km_cirrus":
        bgstr = [
            f"{cirrus_top:.4f} 0.0    0.0",
            f"{cirrus_bottom:.4f} {cirrus_iwc:.4f} {cirrus_eff_radius:.4f}",
        ]
        # no overlap cases
        if bottom > cirrus_top:
            cloudstr = cloudstr + bgstr
        elif top < cirrus_bottom:
            cloudstr = bgstr + cloudstr
        # overlap cases
        else:
            overlap_bottom = max(cirrus_bottom, bottom)
            overlap_top = min(cirrus_top, top)

            tau_contrail_overlap = get_tau(
                iwc=iwc, depth=overlap_top - overlap_bottom, eff_radius=eff_radius
            )
            tau_cirrus_overlap = get_tau(
                iwc=cirrus_iwc,
                depth=overlap_top - overlap_bottom,
                eff_radius=cirrus_eff_radius,
            )

            current_layer = "none"
            # work out the layers
            cloudstr = []

            # enter bottom layer
            if bottom < cirrus_bottom:
                # some contrail below background cirrus start
                cloudstr = [f"{bottom:.4f} {iwc:.4f} {eff_radius:.4f}"] + cloudstr
                current_layer = "contrail"
            elif overlap_bottom > cirrus_bottom:
                # some background cirrus below contrail
                cloudstr = bgstr[1:] + cloudstr
                current_layer = "cirrus"

            # enter overlap layer
            dominant_layer = (
                "contrail" if tau_contrail_overlap > tau_cirrus_overlap else "cirrus"
            )
            if dominant_layer == "contrail" and current_layer != "contrail":
                # contrail dominant overlap layer
                cloudstr = [
                    f"{overlap_bottom:.4f} {iwc:.4f} {eff_radius:.4f}"
                ] + cloudstr
                current_layer = "contrail"
            elif dominant_layer == "cirrus" and current_layer != "cirrus":
                # background cirrus dominant overlap layer
                cloudstr = [
                    f"{overlap_bottom:.4f} {cirrus_iwc:.4f} {cirrus_eff_radius:.4f}"
                ] + cloudstr
                current_layer = "cirrus"

            # exit overlap layer
            dominant_layer = "contrail" if top > cirrus_top else "cirrus"
            if top == cirrus_top:
                dominant_layer = "none"
            if dominant_layer == "cirrus" and current_layer != "cirrus":
                # return to background cirrus
                cloudstr = [
                    f"{overlap_top:.4f} {cirrus_iwc:.4f} {cirrus_eff_radius:.4f}"
                ] + cloudstr
                # cloudstr = bgstr[:1] + cloudstr
            elif dominant_layer == "contrail" and current_layer != "contrail":
                # some contrail above background cirrus end
                cloudstr = [f"{overlap_top:.4f} {iwc:.4f} {eff_radius:.4f}"] + cloudstr

            # exit top layer
            if dominant_layer == "contrail":
                cloudstr = [f"{top:.4f} 0.0 0.0"] + cloudstr
            else:
                cloudstr = [f"{cirrus_top:.4f} 0.0 0.0"] + cloudstr

    cloudstr = "\n".join(cloudstr)
    tmpcloud = tempfile.NamedTemporaryFile(
        delete=False, dir=Path(__file__).parent / "tempfiles"
    )
    tmpcloud.write(cloudstr.encode("ascii"))
    tmpcloud.close()
    return tmpcloud.name


def scene_rt_kwargs(ic_file=None, eff_radius=20.0, atmosphere_file="us", hour=12):
    """
    Generate keyword arguments for running the radiative transfer model for a scene.

    Parameters:
        ic_file: Path to the ice cloud file (optional)
        eff_radius: Effective radius of the cloud particles (microns)
        atmosphere_file: Name of the atmosphere file (default: "us")
        hour: Hour of the day (default: 12)

    Returns:
        rt_kwargs: Dictionary of keyword arguments for the radiative transfer model
    """
    rt_kwargs = {}
    if ic_file is not None:
        rt_kwargs["ic_file"] = f'1D "{ic_file}"'
        if eff_radius > 5:
            rt_kwargs["ic_properties"] = "yang2013 interpolate"
            rt_kwargs["ic_habit_yang2013"] = "droxtal smooth"
        else:
            ic_properties_file = (
                Path(__file__).parent / "data" / "ic.gamma_001.0.mie.cdf"
            )
            rt_kwargs["ic_properties"] = f"{ic_properties_file} interpolate"
    rt_kwargs["atmosphere_file"] = str(Atmosphere.from_name(atmosphere_file).afgl_file)
    rt_kwargs["latitude"] = f"N {LATITUDES[atmosphere_file]}"
    rt_kwargs["longitude"] = f"E {LONGITUDES[atmosphere_file]}"
    rt_kwargs["time"] = f"2023 {DATES[atmosphere_file]} {hour:02} 00 00"
    return rt_kwargs


def brightness_temperature(
    wvl=None, wvn=None, nu=None, radiance_wvl=None, radiance_wvn=None, radiance_nu=None
):
    """
    Compute the brightness temperature.

    Parameters:
        wvl: Wavelength (microns)
        wvn: Wavenumber (cm^-1)
        nu: Frequency (Hz)
        radiance_wvl: Radiance with respect to wavelength (W/m^2/sr/micron)
        radiance_wvn: Radiance with respect to wavenumber (W/m^2/sr/cm^-1)
        radiance_nu: Radiance with respect to frequency (W/m^2/sr/Hz)

    Returns:
        brightness_temp: Brightness temperature (K)
    """
    from scipy.constants import c, h, k

    if [wvl, wvn, nu].count(None) != 2:
        raise ValueError("Exactly one of wvl, wvn, nu must be specified")
    if [radiance_wvl, radiance_wvn, radiance_nu].count(None) != 2:
        raise ValueError(
            "Exactly one of radiance_wvl, radiance_wvn, radiance_nu must be specified"
        )

    if nu is None:
        if wvl is not None:
            nu = c / wvl
        elif wvn is not None:
            nu = c * wvn

    if radiance_nu is None:
        if radiance_wvl is not None:
            radiance_nu = radiance_wvl * c / nu**2
        elif radiance_wvn is not None:
            radiance_nu = radiance_wvn / c

    return h * nu / k / np.log(2 * h * nu**3 / c**2 / radiance_nu + 1)


# def get_optical_depth(cwc, eff_radius, depth, method="k"):
#     """
#     Compute the optical depth of a cloud.

#     Currently unused.

#     Parameters:
#     - cwc: Cloud water content (g/m^3)
#     - eff_radius: Effective radius of the cloud particles (microns)
#     - depth: Thickness of the cloud (km)
#     - method: Method for computing optical depth (default: "k")

#     Returns:
#     - optical_depth: Optical depth of the cloud (km^-1 * km)
#     """
#     cloud = Cloud(LIBRADTRAN_FOLDER, cwc, eff_radius)
#     prop = cloud.cldprp(550, method=method)
#     return prop.ext * depth
