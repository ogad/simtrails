from simtrails.radiative_transfer import (
    SOLAR_CHANNELS,
    brightness_temperature,
    get_ic_file,
    get_output,
    # get_optical_depth,
    scene_rt_kwargs,
)

import os
import itertools
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
import logging


def generate_lookup(
    iwp,
    altitude=[11.3],
    depth=[0.7],
    eff_radius=[20],
    reptran_channel=[
        f"goes-r_abi_ch{i:02d}" for i in [8, 9, 10, 11, 12, 13, 14, 15, 16]
    ],
    albedo=[0.0],
    atmosphere_file=["us"],
    hour=[12],
    background=["Clear"],
    for_rf=False,
    debug=False,
):
    lookup_coords = {
        "iwp": iwp,
        "altitude": altitude,
        "depth": depth,
        "eff_radius": eff_radius,
        "reptran_channel": reptran_channel,
        "albedo": albedo,
        "atmosphere_file": atmosphere_file,
        "hour": hour,
        "background": background,
    }

    if for_rf:
        rf_lookup = xr.DataArray(dims=lookup_coords.keys(), coords=lookup_coords)
        rf_lookup.attrs["units"] = "W/m2"

        od_lookup = xr.DataArray(dims=lookup_coords.keys(), coords=lookup_coords)
        od_lookup.attrs["units"] = ""
    else:
        bt_lookup = xr.DataArray(dims=lookup_coords.keys(), coords=lookup_coords)
        radiance_lookup = xr.DataArray(dims=lookup_coords.keys(), coords=lookup_coords)

        bt_lookup.attrs["units"] = "K"
        radiance_lookup.attrs["units"] = "W/(m^2 nm sr)"

    n_iter = np.prod([len(v) for v in lookup_coords.values()])

    i = 0
    for lookup_point in tqdm(itertools.product(*lookup_coords.values()), total=n_iter):
        (
            iwp_value,
            altitude_value,
            depth_value,
            eff_radius_value,
            reptran_channel_value,
            albedo_value,
            atmosphere_file_value,
            hour_value,
            background_value,
        ) = lookup_point
        if debug:
            if i % 1000 == 0:
                logging.debug(f"GENERATING: i={i}")
            i += 1
        lookup_point_dict = dict(zip(lookup_coords.keys(), lookup_point))

        source_type = "solar" if reptran_channel_value in SOLAR_CHANNELS else "thermal"

        iwc_value = iwp_value / (depth_value * 1e3)
        ic_file = get_ic_file(
            iwc_value,
            altitude_value,
            depth_value,
            eff_radius_value,
            background=background_value,
        )

        rt_kwargs = scene_rt_kwargs(
            ic_file, eff_radius_value, atmosphere_file_value, hour_value
        )
        rt_kwargs |= {
            "mol_abs_param": f"reptran_channel {reptran_channel_value}",
            "source": "solar" if reptran_channel_value in SOLAR_CHANNELS else "thermal",
        }

        if for_rf:
            # Depricated and broken
            rt_kwargs["mol_abs_param"] = "KATO2"
            rt_kwargs["source"] = "thermal"
            rt_kwargs["rte_solver"] = "twostr"
            rt_kwargs["output_process"] = "sum"
            rt_kwargs["output_user"] = "lambda edir edn eup"

            data_thermal = get_output(brightness=False, **rt_kwargs)
            rt_kwargs["source"] = "solar"
            data_solar = get_output(brightness=False, **rt_kwargs)

            forcing = (
                data_solar[1]
                + (data_solar[2] + data_thermal[2])
                - (data_solar[3] + data_thermal[3])
            )  # W/m2

            rf_lookup.loc[lookup_point_dict] = forcing

            # od = get_optical_depth(iwc_value, eff_radius_value, depth_value)
            od_lookup.loc[lookup_point_dict] = od

        else:
            data = get_output(brightness=False, **rt_kwargs)

            radiance_value = data.uu.sel(umu=1).values

            if source_type == "solar":
                # radiance units are W/(m^2 nm sr), dI/dlambda
                radiance_value *= 1e9  # W/(m^2 m sr), dI/dlambda
                bt_value = brightness_temperature(
                    wvl=data.wvl.values * 1e-9, radiance_wvl=radiance_value
                )
            else:
                # radiance units are W/(m^2 sr cm^-1), dI/dwvn
                radiance_value *= 1e-2
                bt_value = brightness_temperature(
                    wvl=data.wvl.values * 1e-9, radiance_wvn=radiance_value
                )

            bt_lookup.loc[lookup_point_dict] = bt_value.item()
            if source_type == "solar":
                radiance_lookup.loc[lookup_point_dict] = (
                    radiance_value.item() * (data.wvl.item()) ** 2 * 1e-11
                )
            else:
                radiance_lookup.loc[lookup_point_dict] = radiance_value.item()

        if ic_file is not None:
            Path(ic_file).unlink()

    if for_rf:
        return xr.Dataset({"rf": rf_lookup, "tau": od_lookup})
    return xr.Dataset({"bt": bt_lookup, "radiance": radiance_lookup})


def expand_lookup_table(lookup_table, dim, new_values):
    existing_coords = {
        coord: lookup_table[coord].values for coord in lookup_table.coords
    }
    new_table = generate_lookup(**(existing_coords | {dim: np.atleast_1d(new_values)}))
    return xr.concat([lookup_table, new_table], dim=dim)


if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.realpath(__file__))
    lookup_table = generate_lookup(
        iwp=np.linspace(0, 100, 30_001),
        altitude=[8.3, 9.3, 10.3, 11.3, 12.3],
    )
    lookup_table.to_netcdf(os.path.join(data_dir, "lookup.nc"))
