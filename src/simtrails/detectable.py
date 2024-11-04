from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List

from abc import ABC, abstractmethod
import numpy as np


if TYPE_CHECKING:
    from .imager import Imager


class Detectable(ABC):
    """
    Abstract base class for detectable objects.

    This class defines the common interface and behavior for detectable objects.
    It carries the infrastructure to convert use the grid values to look up the
    radiative transfer outcome.
    
    Subclasses must implement the abstract methods and properties defined here.

    Required child class attributes:
        - grid (np.ndarray): The grid of values for the detectable object.
        - grid_resolution (float): The resolution of the grid in kilometers.
        - hour (int): The hour of the contrail generation in XX:00 format.
        - atmosphere_file (str): The file name of the atmosphere data.

    """

    def __post_init__(self, generate_lookup=True):
        """Initialise the lookup table so real-time simulation can be skipped."""
        self.lookup_decimals = 4
        self._lookup = None if generate_lookup else False

    @property
    def lookup(self):
        """The lookup table of radiative transfer properties."""
        if self._lookup == False:
            return None
        if self._lookup is not None:
            return self._lookup

        self._lookup = self.fetch_lookup(self.lookup_decimals)  # to move to iwps
        return self._lookup

    def clear_memory(self):
        """Clear the lookup table from memory."""
        self._lookup = None if self._lookup is not False else self._lookup
        self._grid = None

    @abstractmethod
    def rt_kwargs(self, grid_value) -> dict:
        """Return the keyword arguments for the real-time simulation based on the given grid value."""
        ...

    @abstractmethod
    def _iwp(self, grid_value) -> float:
        """Calculate the ice water content (IWP) based on the given grid value."""
        ...

    @abstractmethod
    def _pixel_lookup_key(self, grid_value) -> dict:
        """Return the partial lookup table key for the given grid value."""
        ...

    def get_rt_output(
        self, imager: Imager, i_channel: int, shape: tuple[int, int] = None
    ) -> np.ndarray:
        """
        Get the radiative transfer output for the detectable object.

        Args:
            imager (Imager): The imager object used for radiative transfer calculations.
            i_channel (int): The channel number to calculate the output for.
            shape (tuple[int, int], optional): The desired shape of the output grid. If not provided,
                the shape of the detectable object's grid will be used.

        Returns:
            np.ndarray: The radiative transfer output as a numpy array.

        Raises:
            ValueError: If the provided shape is smaller than the detectable object's grid shape.
        """
        import numpy as np

        # TODO: the logic allows for 2D grids; if we want more properties than just
        # contrail flag, we'll need to change this. We could try calculating
        # each grid point (i.e. only the first two indices) a third dimension.

        # Get the grid, resizing if necessary; not normally used any more, but
        # left in case we want to reactivate
        if shape is not None:
            grid_shape = self.grid.shape
            if grid_shape[0] > shape[0] or grid_shape[1] > shape[1]:
                raise ValueError(
                    f"Detectable grid shape {grid_shape} is larger than prescribed shape {shape}"
                )
            grid = np.zeros(shape)
            grid[: self.grid.shape[0], : self.grid.shape[1]] = self.grid
        else:
            grid = None

        bt = self.output_from_values(imager, i_channel, output_grid=grid)

        return bt

    def output_from_values(
        self, imager: Imager, i_channel: int, output_grid: np.ndarray = None
    ):
        """
        Calculate the output values based on the input grid, imager, and channel index.

        Parameters:
        grid (np.array): The input grid.
        imager (Imager): The imager object.
        i_channel (int): The channel number.

        Returns:
        np.array: The calculated output values.
        """

        grid = output_grid if output_grid is not None else self.grid

        if self.lookup is not None:
            if i_channel == 0:
                bt = grid
            else:
                masked_grid = np.ma.masked_array(grid, grid == 0)
                bt = np.interp(masked_grid, self.lookup[0], self.lookup[i_channel - 12])
                bt = np.ma.filled(bt, fill_value=self.lookup[i_channel - 12][0])
        else:
            rounded_grid = np.round(grid, self.lookup_decimals)
            unique_values = np.unique(rounded_grid)
            vfunc = np.vectorize(
                lambda x: self.simulate_pixel_radiance(x, imager, i_channel)
            )
            bt_values = vfunc(unique_values)
            bt = bt_values[
                np.where(unique_values == rounded_grid[:, :, None])[-1]
            ].reshape(rounded_grid.shape)

        return bt

    def simulate_pixel_output(self, grid_value: float, imager: Imager, i_channel: int):
        """
        Simulates the pixel output for a given grid value, imager, and channel.

        This is only used for the real-time simulation, and is not invoked if
        a lookup table is provided.

        Args:
            grid_value (float): The value of the grid.
            imager (Imager): The imager object.
            i_channel (int): The channel number.

        Returns:
            data (xr.Dataset): The simulated pixel output.

        Raises:
            ValueError: If an error occurs during the simulation.
        """
        from .radiative_transfer import get_output
        from pathlib import Path

        rt_kwargs = self.rt_kwargs(grid_value) | imager.get_rt_options(i_channel)
        rt_kwargs = {k: v for k, v in rt_kwargs.items() if v is not None}

        try:
            data = get_output(**rt_kwargs)
        except ValueError:
            print(rt_kwargs)
            raise

        if "ic_file" in rt_kwargs.keys():
            Path(rt_kwargs["ic_file"].split()[1][1:-1]).unlink()

        return data

    def simulate_pixel_radiance(
        self, grid_value: float, imager: Imager, i_channel: int
    ) -> float:
        """
        Simulates the pixel radiance for a given grid value, imager, and channel.

        Parameters:
        grid_value (float): The value of the grid.
        imager (Imager): The imager object.
        i_channel (int): The channel number.

        Returns:
        float: The simulated pixel radiance.
        """

        if i_channel == 0:
            return grid_value
        data = self.simulate_pixel_output(grid_value, imager, i_channel)
        radiance_bt = data.uu.sel(umu=1).item()
        return radiance_bt

    def pixel_optical_depth(
        self, grid_value: float, imager: Imager, i_channel: int
    ) -> float:
        """
        Calculate the pixel optical depth for a given grid value, imager, and channel.

        Parameters:
        grid_value (float): The value of the grid.
        imager (Imager): The imager object.
        i_channel (int): The channel number.

        Returns:
        float: The pixel optical depth.

        Raises:
        NotImplementedError: This method is not currently implemented.
        """
        data = self.simulate_pixel_output(grid_value, imager, i_channel)
        raise NotImplementedError(
            "TODO: implement optical depth calculation without using verbose output"
        )
        # if len(verb["optprop"].wvl) > 1:
        #     raise ValueError(
        #         "RT output has more than one wavelength; use a detector band."
        #     )
        # optical_depth = -1 * np.trapz(
        #     verb["optprop"].ic_abs + verb["optprop"].ic_sca, verb["optprop"].z
        # )
        # return optical_depth

    def max_optical_depth(self, imager: Imager, i_channel: int) -> float:
        """
        Get the maximum optical depth for a given imager and channel.

        Parameters:
        imager (Imager): The imager object.
        i_channel (int): The channel number.

        Returns:
        float: The maximum optical depth.

        Raises:
        NotImplementedError: This method is not currently implemented.
        """
        return self.pixel_optical_depth(self.grid.max(), imager, i_channel)[0]

    def fetch_lookup(self, n_decimals: int) -> Optional[List[np.ndarray]]:
        """
        Fetch the lookup values for the given number of decimals.

        Parameters:
        n_decimals (int): The number of decimals.

        Returns:
        Optional[List[np.ndarray]]: The lookup values or None if an error occurs.
        """
        from simtrails.lookup import lookup_interp

        n_points = 10**n_decimals + 1

        grid_values = np.round(np.linspace(0, 1, n_points), n_decimals)
        iwps = self._iwp(grid_values)  # to move to iwps
        lookups = []
        lookups.append(grid_values)
        lookup_key = {
            "albedo": 0,
            "atmosphere_file": self.atmosphere_file,
            "hour": self.hour,
        } | self._pixel_lookup_key(grid_values[0])
        # lookup_key.pop("iwc")
        lookup_key["iwp"] = iwps
        for i, channel in enumerate([13, 14, 15]):
            lookup_key["reptran_channel"] = f"goes-r_abi_ch{channel:02d}"
            try:
                lookups.append(lookup_interp(**lookup_key))
            except ValueError as e:
                print(e)
                return None
        return lookups


class EmptyScene(Detectable):
    def __init__(
        self,
        size: tuple[int, int],
        resolution: float = 0.25,
        albedo: float = 0,
        atmosphere_file: str = None,
        hour: int = 12,
    ):
        """
        Initialize the EmptyScene object.

        Parameters:
        size (tuple[int, int]): The size of the scene.
        resolution (float): The grid resolution. Default is 0.25.
        albedo (float): The albedo value. Default is 0.
        atmosphere_file (str): The atmosphere file. Default is None.
        hour (int): The hour of the scene. Default is 12.
        """
        self.size = size
        self.grid = np.zeros(size)
        self.grid_resolution = resolution
        self.albedo = albedo
        self.atmosphere_file = "us" if atmosphere_file is None else atmosphere_file
        self.hour = hour

    def rt_kwargs(self, _) -> dict:
        """
        Get the radiative transfer keyword arguments.

        Parameters:
        _ : Unused parameter.

        Returns:
        dict: The radiative transfer keyword arguments.
        """
        return {"albedo": self._albedo}

    @property
    def lookup_id(self) -> str:
        """
        Get the lookup ID.

        Returns:
        str: The lookup ID.
        """
        return f"EmptyScene({self.size}, {self.resolution}, {self.albedo})"

    def _pixel_lookup_key(self, _) -> dict:
        """
        Get the pixel lookup key.

        Parameters:
        _ : Unused parameter.

        Returns:
        dict: The pixel lookup key.
        """
        return {
            "iwp": 0,
            "altitude": 11.3,
            "depth": 0.7,
            "eff_radius": 20,
        }


class TestPixel(Detectable):
    def __init__(
        self,
        altitude: float = 11.0,  # km
        depth: float = 0.5,  # km
        iwp: float = None,  # g/m^2
        iwc: float = None,  # g/m^3; default is 0.01 g/m^3
        n_ice: float = None,  # L-1; uses spherical
        eff_radius: float = None,  # um default is 20 um
        atmosphere_file: str = "us",
        hour: int = 12,  # XX:00
        generate_lookup: bool = False,
    ):
        """
        Initialize the TestPixel object.

        Parameters:
        base (float): The base value in kilometers. Default is 11.0.
        depth (float): The depth value in kilometers. Default is 0.5.
        iwp (float): The ice water path value in g/m^2. Default is None.
        iwc (float): The ice water content value in g/m^3. Default is None.
        n_ice (float): The ice particle concentration value in L-1. Default is None.
        eff_radius (float): The effective radius value in um. Default is None.
        atmosphere_file (str): The atmosphere file. Default is "us".
        hour (int): The hour of the scene. Default is 12.
        generate_lookup (bool): Whether to generate the lookup table. Default is False.
        """
        from simtrails.misc.microphysics import ic_lrt_properties

        iwp, eff_radius = ic_lrt_properties(iwc, iwp, eff_radius, n_ice, depth)

        self.iwp: float = np.round(iwp, 4)  # g/m^3
        self.depth: float = depth  # km
        self.altitude: float = altitude  # km
        self.eff_radius: float = eff_radius  # um

        self.grid = np.array([[0]])
        self.grid_resolution = None
        self.atmosphere_file = "us" if atmosphere_file is None else atmosphere_file
        self.hour = hour

        super().__post_init__(generate_lookup=generate_lookup)

    def rt_kwargs(self, _) -> dict:
        """
        Get the radiative transfer keyword arguments.

        Parameters:
        _ : Unused parameter.

        Returns:
        dict: The radiative transfer keyword arguments.
        """
        from simtrails.radiative_transfer import get_ic_file, scene_rt_kwargs

        iwc_value = self.iwp / (self.depth * 1e3)
        ic_file = get_ic_file(iwc_value, self.altitude, self.depth, self.eff_radius)
        rt_kwargs = scene_rt_kwargs(
            ic_file, self.eff_radius, self.atmosphere_file, self.hour
        )

        return rt_kwargs

    def _iwp(self, _) -> float:
        """
        Get the ice water content.

        Parameters:
        _ : Unused parameter.

        Returns:
        float: The ice water content.
        """
        return self.iwp

    def _pixel_lookup_key(self, _) -> dict:
        """
        Get the pixel lookup key.

        Parameters:
        _ : Unused parameter.

        Returns:
        dict: The pixel lookup key.
        """
        return {
            "iwp": self.iwp,
            "altitude": self.altitude,
            "depth": self.depth,
            "eff_radius": self.eff_radius,
        }
