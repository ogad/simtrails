# GOES satellite bands in LRT...

from ..radiative_transfer import get_output


def get_output_cloud_phases(channels=range(7, 17), **kwargs):  # TODO: decomission
    import numpy as np
    import xarray as xr

    clear_outputs = []
    ic_outputs = []
    wc_outputs = []
    for i_channel in channels:
        data = get_output(
            source="thermal",
            ic_file="1d /Users/ogd22/Library/CloudStorage/OneDrive-ImperialCollegeLondon/10_Exploratory/pylrt_test/ic_test.dat",
            mol_abs_param=f"reptran_channel goes-r_abi_ch{i_channel:02d}",
            **kwargs,
        )
        ic_outputs.append(data)

        data = get_output(
            source="thermal",
            mol_abs_param=f"reptran_channel goes-r_abi_ch{i_channel:02d}",
            **kwargs,
        )
        clear_outputs.append(data)

        water_cloud = {
            "z": np.array([4, 3.7]),
            "lwc": np.array([0, 0.5]),
            "re": np.array([0, 20]),
        }
        data = get_output(
            cloud=water_cloud,
            source="thermal",
            mol_abs_param=f"reptran_channel goes-r_abi_ch{i_channel:02d}",
            **kwargs,
        )
        wc_outputs.append(data)

    clear_data = xr.concat(clear_outputs, dim="wvl")
    clear_data = clear_data.assign_coords(channel=("wvl", channels))
    ic_data = xr.concat(ic_outputs, dim="wvl")
    ic_data = ic_data.assign_coords(channel=("wvl", channels))
    wc_data = xr.concat(wc_outputs, dim="wvl")
    wc_data = wc_data.assign_coords(channel=("wvl", channels))

    return clear_data, ic_data, wc_data


def get_btd(ds, data_var="eup"):
    import xarray as xr

    btds = {}

    if "channel" not in ds.dims:
        ds = ds.swap_dims({"wvl": "channel"})

    for ch1 in ds.channel.values:
        btds[ch1] = (ds.sel(channel=ch1)[data_var].values) - ds[data_var]
    da_btd = xr.concat(btds.values(), dim="ch1")
    da_btd = da_btd.rename({"channel": "ch2", "wvl": "wvl2"})
    da_btd = da_btd.assign_coords(
        ch1=ds.channel.values, wvl1=("ch1", da_btd.wvl2.values)
    )

    return da_btd
