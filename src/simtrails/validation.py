import logging
import numpy as np


def check_bounding(observability_dataset, force_pass=False):

    # (var, check unobservable @ low, check unobservable @ high)
    # where true, var must be unobservable OR there is no CoCiP data near that extreme
    checks = [
        ("iwp", True, False),
        ("eff_radius", False, True),
        ("width", True, True),
    ]
    for check in checks:
        make_check(observability_dataset, *check, force_pass=force_pass)


def make_check(observability_dataset, dim, check_low, check_high, force_pass=False):
    if dim not in observability_dataset.dims:
        return

    obs_low, obs_high = data_at_extreme(observability_dataset, dim)

    fails_obs_low = obs_low and check_low
    fails_obs_high = obs_high and check_high

    if fails_obs_high or fails_obs_low:
        # still observable at an extreme that should be unobservable
        # check for CoCiP data
        hist_vars = [v for v in observability_dataset.data_vars if v != "observability"]
        for var in hist_vars:
            data_low, data_high = data_at_extreme(observability_dataset, dim, var)
            if fails_obs_high and data_high and fails_obs_low and data_low:
                if data_high > 0.01 or data_low > 0.01 and not force_pass:
                    raise ValueError(
                        f"Still observable (and CoCiP {var} data) at both extremes of {dim}. ({data_low:.1%} - {data_high:.1%} of data)"
                    )
                else:
                    logging.warning(
                        f"Still observable (and CoCiP {var} data) at both extremes of {dim}. (<1% of data: {data_low:.1%} - {data_high:.1%})"
                    )
            if fails_obs_low and data_low:
                if data_low > 0.01 and not force_pass:
                    raise ValueError(
                        f"Still observable (and CoCiP {var} data) at low {dim}. ({data_low:.1%} of data)"
                    )
                else:
                    logging.warning(
                        f"Still observable (and CoCiP {var} data) at low {dim}. (<1% of data: {data_low:.1%})"
                    )
            if fails_obs_high and data_high:
                if data_high > 0.01 and not force_pass:
                    raise ValueError(
                        f"Still observable (and CoCiP {var} data) at high {dim}. ({data_high:.1%} of data)"
                    )
                else:
                    logging.warning(
                        f"Still observable (and CoCiP {var} data) at high {dim}. (<1% of data: {data_high:.1%})"
                    )


def data_at_extreme(observability_dataset, dim, var="observability"):
    if dim not in observability_dataset.dims:
        return
    var_dim = (
        observability_dataset[var]
        .sum([d for d in observability_dataset.dims if d != dim])
        .sortby(dim)
        .values
    )

    # if var == "observability":
    #     total = observability_dataset["observability"].sum()
    #     var_dim = var_dim[1:-1]
    # else:
    #     total = observability_dataset[var].attrs["total"]
    data_high = var_dim[-1] / var_dim.sum()
    data_low = var_dim[0] / var_dim.sum()
    return data_low, data_high


def validated_contrail_mask(mask, orientation=np.pi / 3):
    from skimage import measure

    if not mask.any():
        return mask

    numbered_blobs = measure.label(mask)
    regions = measure.regionprops(numbered_blobs)

    region = max(regions, key=lambda r: r.area)

    if np.abs(region.orientation - orientation) > np.pi / 6:
        # bad mask
        mask = np.zeros_like(mask, dtype=mask.dtype)
    return mask
