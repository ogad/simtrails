import numpy as np
import xarray as xr


import os

location = os.path.dirname(os.path.realpath(__file__))
lookup_file = os.path.join(location, "data", "lookup_table.nc")

RENAME_DICT = {
    "base": "altitude",
    "peak_iwc": "iwc",
    "thickness": "depth",
    "peak_iwp": "iwp",
}

_LOOKUP = None


def load_lookup():
    lookup = xr.load_dataset(lookup_file)
    lookup_rename_dict = {k: v for k, v in RENAME_DICT.items() if k in lookup}
    if len(lookup_rename_dict) > 0:
        lookup = lookup.rename(lookup_rename_dict)
    return lookup


def lookup_interp(lookup_table="default", var="bt", **kwargs):
    """
    Interpolates values from a lookup table based on provided coordinates.

    Args:
        lookup_table (xarray.Dataset): The lookup table containing the data.
        var (str): The variable to interpolate from the lookup table.
        **kwargs: Keyword arguments representing the coordinates for interpolation.

    Returns:
        xarray.DataArray: The interpolated values.

    Raises:
        ValueError: If any required coordinates are missing or if any lookup values are invalid or out of range.
    """
    global _LOOKUP
    if lookup_table == "default":
        if _LOOKUP is None:
            _LOOKUP = load_lookup()
        lookup_table = _LOOKUP

    required_coords = [coord for coord in lookup_table.coords if coord not in kwargs]
    if len(required_coords) > 0:
        raise ValueError(
            f"Missing required lookup table coordinates: {required_coords}"
        )

    str_coords = {}
    for coord, value in kwargs.items():
        if type(value) is str:
            if value not in lookup_table[coord]:
                raise ValueError(
                    f"Lookup value {value} for coordinate {coord} is not valid"
                )
            else:
                str_coords[coord] = value
        elif (
            np.array(value < lookup_table[coord].min().item()).any()
            or np.array(value > lookup_table[coord].max().item()).any()
        ):
            raise ValueError(
                f"Lookup value {value} for coordinate {coord} is out of range"
            )

    num_coords = {
        coord: value for coord, value in kwargs.items() if coord not in str_coords
    }

    values = lookup_table[var].sel(**str_coords)
    squeezed_dims = []
    for k, v in num_coords.items():
        if len(values[k]) == 1:
            if v != values[k].item():
                raise ValueError(f"Lookup value {v} for coordinate {k} is out of range")
            values = values.squeeze(k)
            squeezed_dims.append(k)

    num_coords = {k: v for k, v in num_coords.items() if k not in squeezed_dims}

    return values.interp(**num_coords)
