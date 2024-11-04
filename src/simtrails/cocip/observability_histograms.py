# %%
from copy import deepcopy
import numpy as np
import xarray as xr


def get_observability(resolution, overall_result, plot=False):
    result = deepcopy(overall_result)
    result.data = result.data.sel(resolution=resolution)

    if plot:
        result.plot_2d(*result.data.dims)

    # observability = result.data["area"].mean("repeat").squeeze()
    observability = result.data["observability"].squeeze()
    hist_obs = observability.values

    # need to reverse the dims and values because of the way the histograms are made
    dims = list(observability.dims)
    values = [observability[i].values for i in dims]

    return hist_obs, dims, values


def observable_fraction(
    observability_dataset,
    category,
    observability_threshold=0.5,
    pc_of_max=False,
    fill_from_min=False,
):
    observability = observability_dataset["observability"]
    hist = observability_dataset[category]

    hist = np.abs(hist)
    observable = find_observable(
        observability,
        observability_threshold=observability_threshold,
        fill_from_min=fill_from_min,
        pc_of_max=pc_of_max,
    )

    pc_obs = np.nansum(np.abs((hist * observable)) / hist.sum())

    # error analysis
    high_observable = find_observable(
        observability,
        observability_threshold=0.05,
        fill_from_min=fill_from_min,
        pc_of_max=pc_of_max,
    )
    low_observable = find_observable(
        observability,
        observability_threshold=0.95,
        fill_from_min=fill_from_min,
        pc_of_max=pc_of_max,
    )
    high_pc_obs = np.nansum(
        np.abs((hist * high_observable)) / hist.sum()
    )  # nanvalues @ overflow edges.
    low_pc_obs = np.nansum(np.abs((hist * low_observable)) / hist.sum())

    plus_error = max(high_pc_obs - pc_obs, 0)
    minus_error = max(pc_obs - low_pc_obs, 0)
    return pc_obs, (minus_error, plus_error)


def find_observable(
    observability: xr.DataArray,
    observability_threshold: float = 0.5,
    fill_from_min=False,
    pc_of_max=False,
):
    if not pc_of_max:
        observable = observability > observability_threshold
    else:
        observable = observability > observability.max() * observability_threshold

    if fill_from_min:
        directions = {
            "iwp": 1,  # fill from low
            "eff_radius": -1,  # fill from high
            "width": 1,  # fill from low
        }
        # replace any "false" values with "true" if there is a "true" preceeding it
        for dim, direction in directions.items():
            if direction == -1:
                observable = observable.sortby(dim, ascending=False)
            # work out cumulative sum in the direction of the extreme
            cumulative_observable = observable.cumsum(dim=dim)
            observable = cumulative_observable > 0
    return observable


# %%
if __name__ == "__main__":
    from simtrails.cocip.data import CocipDataset

    histograms = CocipDataset.from_pq("12:00").observability_dataset(
        sensitivity={
            "iwp": np.logspace(-3, 1.5, 41),  # quicker
            "eff_radius": np.logspace(np.log10(1), np.log10(50), 40),  # quicker
            "resolution": np.array([0.5]),
        },
        resolution=0.5,
        contrail_parameters={"width": 2},
        n_repeats=4,
        n_proc=4,
    )
# %%
