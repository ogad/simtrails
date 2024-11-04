from __future__ import annotations
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING
from numpy.typing import ArrayLike
import numpy as np

if TYPE_CHECKING:
    from simtrails.detectable import Detectable


class Imager:
    """
    The Imager class represents an imager used for simulating observations of detectable objects.

    Attributes:
        channels (dict[int, ImagerChannel]): A dictionary of imager channels, where the key is the channel index
            and the value is an instance of ImagerChannel.
    """

    def __init__(self, channels: dict[int, ImagerChannel]):
        self.channels = channels

    @classmethod
    def from_name(cls, name: str) -> Imager:
        """
        Returns an instance of Imager based on the given name.

        Args:
            name (str): The name of the imager.

        Returns:
            Imager: An instance of Imager.

        Raises:
            NotImplementedError: If the imager with the given name is not implemented.
        """
        if name == "GOES_R_ABI":
            return GOES_R_ABI
        else:
            raise NotImplementedError(f"Imager {name} not implemented")

    def simulate_observation(
        self, detectable: Detectable, i_channel: int, pbar=None, shape=None
    ) -> ArrayLike:
        """
        Simulates an observation of a detectable object using the specified imager channel.

        Args:
            detectable (Detectable): The detectable object to simulate the observation for.
            i_channel (int): The number of the imager channel to use for the observation; i_channel=0 corresponds to
                the raw scene grid values.
            pbar (optional): A progress bar object to update during the simulation.
            shape (optional): The desired shape of the output observation grid.

        Returns:
            ArrayLike: The simulated observation data.

        Raises:
            ValueError: If the forced shape is smaller than the output grid.
        """
        rt_output = detectable.get_rt_output(self, i_channel)

        data_output_grid = self.channels[i_channel].regrid_observable(
            rt_output, detectable.grid_resolution
        )

        if shape is not None:
            if (
                shape[0] < data_output_grid.shape[0]
                or shape[1] < data_output_grid.shape[1]
            ):
                raise ValueError(
                    f"Forced shape {shape} is smaller than the output grid {data_output_grid.shape}"
                )

            bg_value = detectable.output_from_values(0, self, i_channel)
            resized_grid = np.ones(shape) * bg_value
            resized_grid[: data_output_grid.shape[0], : data_output_grid.shape[1]] = (
                data_output_grid
            )
            data_output_grid = resized_grid

        output_with_noise = self.channels[i_channel].apply_noise(data_output_grid)

        if pbar:
            pbar.update()
        return output_with_noise

    def get_rt_options(self, i_channel: int) -> dict:
        """
        Returns the options for radiative transfer simulation for the specified imager channel.

        Args:
            i_channel (int): The index of the imager channel.

        Returns:
            dict: The options for radiative transfer simulation.
        """
        options = copy(vars(self.channels[i_channel]))
        options.pop("noise_equiv_temp_diff")
        options.pop("resolution")
        return options


class GOESImager(Imager):
    """
    The GOESImager class represents a specific imager, the GOES-R ABI.

    The GOESImager class can represent GOES-like satellites, but has variable resolution.

    Attributes:
        resolution (float): The resolution of the imager channels in kilometers.

    Methods:
        __init__(self, resolution=0.5): Initializes a GOESImager instance with the specified resolution.
    """

    def __init__(self, resolution=0.5, nedt=0.03):
        super().__init__(
            {
                i: ImagerChannel("solar", f"reptran_channel goes-r_abi_ch{i:02d}")
                for i in range(7)
            }
            | {
                i: ImagerChannel("thermal", f"reptran_channel goes-r_abi_ch{i:02d}")
                for i in range(7, 17)
            }
        )
        for i_channel in self.channels.keys():
            self.channels[i_channel].resolution = resolution
            self.channels[i_channel].noise_equiv_temp_diff = nedt


@dataclass
class ImagerChannel:
    """
    The ImagerChannel class represents a channel of an imager.

    Attributes:
        source (str): The source (solar/thermal) of the channel.
        mol_abs_param (str): The molecular absorption parameterisation of the channel.
        noise_equiv_temp_diff (float): The noise equivalent temperature difference of the channel in Kelvin.
        resolution (float): The resolution of the channel in kilometers.
    """

    source: str
    mol_abs_param: str
    noise_equiv_temp_diff: float = 0.03  # 0.15  # K
    resolution: float = 0.5  # km
    calib_temp: float = 0.2  # K

    def apply_noise(self, data: ArrayLike) -> ArrayLike:
        """
        Applies noise to the given data.

        Args:
            data (ArrayLike): The data to apply noise to.

        Returns:
            ArrayLike: The data with noise applied.
        """
        import numpy as np

        noise = np.random.normal(0, self.noise_equiv_temp_diff, data.shape)
        calib_error = np.random.normal(0, self.calib_temp)
        return data + noise + calib_error

    def regrid_observable(self, data: ArrayLike, data_resolution: float) -> ArrayLike:
        """
        Regrids the observable data from the detectable grid to the specified resolution.

        Args:
            data (ArrayLike): The observable data to regrid.
            data_resolution (float): The resolution of the data.

        Returns:
            ArrayLike: The regridded data.

        Raises:
            ValueError: If the data resolution is greater than the imager resolution.
        """
        from skimage.transform import rescale
        from skimage.transform import downscale_local_mean

        if data_resolution is None:
            return data

        if data_resolution > self.resolution:
            raise ValueError(
                f"Data resolution {data_resolution} is greater than imager resolution {self.resolution}"
            )

        if data_resolution == self.resolution:
            return data

        rescale_factor = self.resolution / data_resolution
        if int(rescale_factor) != rescale_factor:
            raise ValueError(
                f"Data resolution {data_resolution} is not a multiple of imager resolution {self.resolution}"
            )
            rescale_factor = data_resolution / self.resolution
            return rescale(data, rescale_factor)
        else:
            rescale_factor = int(rescale_factor)
            return downscale_local_mean(data, (rescale_factor, rescale_factor))

    @property
    def reptran_channel(self) -> str:
        """
        Returns the reptran channel of the imager channel.

        Returns:
            str: The reptran channel.
        """
        return self.mol_abs_param.split(" ")[-1]


GOES_R_ABI = GOESImager()
