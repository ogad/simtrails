from simtrails.detectable import Detectable
import numpy as np


class Contrail(Detectable):
    """A contrail object that can be used for simulating contrails in radiative transfer simulations.

        Args:
            length (float): The length of the contrail in kilometers.
            width (float): The width of the contrail in kilometers.
            angle (float): The angle of the contrail in degrees.
            altitude (float): The base altitude of the contrail in kilometers.
            depth (float): The thickness of the contrail in kilometers.
            iwp (float, optional): The peak ice water path of the contrail in g/m^2.
                If not specified, iwc and thickness are used.
            iwc (float, optional): The peak ice water content of the contrail in g/m^3.
            n_ice (float, optional): The number concentration of ice particles in L-1.
                Uses spherical particles.
            resolution (float): The grid resolution in kilometers.
            eff_radius (float, optional): The effective radius of the ice particles in micrometers.
                Default is 20 um.
            atmosphere_file (str): The file name of the atmosphere data.
            hour (int): The hour of the contrail generation in XX:00 format.
            generate_lookup (bool): Whether to generate lookup tables.
            background (str): The background atmosphere to use for the contrail."""

    def __init__(
        self,
        length: float = 150,  # km
        width: float = 2,  # km
        angle: float = 30.0,  # degrees
        altitude: float = 11.0,  # km
        depth: float = 0.5,  # km
        iwp: float = None,  # g/m^2; default is 5 g/m^2
        iwc: float = None,  # g/m^3
        n_ice: float = None,  # L-1; uses spherical (?)
        resolution: float = 0.25,  # km
        eff_radius: float = None,  # um default is 10 um
        atmosphere_file: str = "us",
        hour: int = 12,  # XX:00
        generate_lookup: bool = True,
        background: str = "Clear",
    ) -> None:
        from simtrails.misc.microphysics import ic_lrt_properties

        iwp, eff_radius = ic_lrt_properties(iwc, iwp, eff_radius, n_ice, depth)

        self.grid_resolution: float = resolution  # km
        self.length: float = length  # km
        self.width: float = width  # km
        self.angle: float = angle  # degrees

        self.iwp: float = np.round(iwp, 4)  # g/m^2
        self.depth: float = depth  # km
        self.altitude: float = altitude  # km
        self.eff_radius: float = eff_radius  # um
        self.atmosphere_file: str = atmosphere_file
        self.hour: int = hour
        self.background: str = background

        if (
            self.length / self.grid_resolution > 1000
            or self.width / self.grid_resolution > 1000
        ):
            raise ValueError(
                "Contrail dimensions must be less than 1000 times the grid resolution"
            )
        self._grid: np.ndarray = None

        Detectable.__post_init__(self, generate_lookup=generate_lookup)

    @property
    def grid(self) -> np.ndarray:
        """
        Generate a grid representation of the contrail.

        Returns:
            numpy.ndarray: The grid representation of the contrail.

        Raises:
            ValueError: If the contrail dimensions are not multiples of the grid resolution.
        """
        import numpy as np
        from skimage.transform import rotate
        from scipy.ndimage import gaussian_filter

        if self._grid is not None:
            return self._grid

        if (
            self.width % self.grid_resolution != 0
            or self.length % self.grid_resolution != 0
        ):
            raise ValueError("Contrail dimensions must be multiples of grid resolution")

        # TODO: deal with contrails that aren't multiples of the grid resolution
        width_px = int(np.round(self.width / self.grid_resolution))
        length_px = int(np.round(self.length / self.grid_resolution))

        base_contrail = np.ones((width_px, length_px))
        padded_contrail = np.pad(base_contrail, 32)
        rotated_contrail = rotate(
            padded_contrail, -1 * self.angle, preserve_range=True, resize=True
        )

        extra_padding = 8 - np.array(rotated_contrail.shape) % 8
        rand_offset = np.random.randint(0, 8, 2)
        padding = np.concatenate(
            [
                np.floor(extra_padding / 2) + rand_offset,
                np.ceil(extra_padding / 2) + (8 - rand_offset),
            ]
        ).astype(int)
        rotated_contrail = np.pad(
            rotated_contrail, ((padding[0], padding[2]), (padding[1], padding[3]))
        )

        smoothed_contrail = gaussian_filter(rotated_contrail, 0.6)

        longest_px = max(length_px, width_px)
        contrail_grid = smoothed_contrail

        self._grid = contrail_grid
        self.xy_offset = (32 + padding[:2:-1]) * self.grid_resolution
        return contrail_grid

    def rt_kwargs(self, grid_value: float) -> dict:
        """
        Generate the keyword arguments for radiative transfer simulation of a single pixel.

        Args:
            grid_value (float): The grid value.

        Returns:
            dict: The keyword arguments for scene radiative transfer simulation.
        """
        from .radiative_transfer import scene_rt_kwargs

        ic_file = self.pixel_ic_file(grid_value)
        rt_kwargs = scene_rt_kwargs(
            ic_file, self.eff_radius, self.atmosphere_file, self.hour
        )
        return rt_kwargs

    def pixel_ic_file(self, contrail_flag_value: float) -> str:
        """Generate the temporary ic_file for a single pixel."""
        from .radiative_transfer import get_ic_file

        if contrail_flag_value == 0:
            return None
        else:
            iwc = self.iwp * contrail_flag_value / (self.depth * 1e3)
            return get_ic_file(
                iwc,
                self.altitude,
                self.depth,
                self.eff_radius,
                background=self.background,
            )

    def _pixel_lookup_key(
        self, grid_value: float = None
    ) -> (
        dict
    ):  # WARNING: this needs to only vary on IWP because of the way the lookup table is generated in Detectable.generate_lookup
        key = {
            "altitude": self.altitude,
            "depth": self.depth,
            "eff_radius": self.eff_radius,
            "background": self.background,
        }
        if grid_value is not None:
            key["iwp"] = self._iwp(grid_value)

        return key

    def _iwp(self, grid_value: float) -> float:
        return np.maximum(self.iwp * grid_value, 0.0)

    def plot(self, to_plot: np.ndarray = None, ax=None, **kwargs):
        """Plot the contrail grid."""

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        if to_plot is None:
            to_plot = self.grid

        if "cmap" not in kwargs:
            kwargs["cmap"] = "gray"

        hdl = ax.imshow(to_plot, vmin=0, vmax=1, **kwargs)
        cb = plt.colorbar(hdl, label="Simulated contrail mask (\%)")
        ax.set_aspect("equal")
        return ax

    def __str__(self) -> str:
        return f"Contrail(length={self.length}km, width={1000*self.width}m, thickness={self.depth}km, angle={self.angle}°, base={self.altitude}km, Peak IWP={self.iwp}g/m^2, Reff={self.eff_radius}µm)"

    def __repr__(self) -> str:
        return f"< {str(self)} @ {id(self)} >"


class GaussianContrail(Contrail):
    """A contrail object that uses a Gaussian profile for the IWP across the width of the contrail."""
    @property
    def grid(self) -> np.ndarray:
        """
        Generate a grid representation of the contrail.

        Returns:
            numpy.ndarray: The grid representation of the contrail.

        Raises:
            ValueError: If the contrail dimensions are not multiples of the grid resolution.
        """
        import numpy as np

        angle_quadrant = (self.angle // 90) % 4
        angle = self.angle % 90

        if self._grid is not None:
            return self._grid

        # TODO: deal with contrails that aren't multiples of the grid resolution
        width_px = self.width / self.grid_resolution
        length_px = self.length / self.grid_resolution

        if self.length % self.grid_resolution != 0:
            raise ValueError("Contrail length must be multiples of grid resolution")

        uvec_line = np.array(
            [np.sin(np.deg2rad(self.angle)), np.cos(np.deg2rad(self.angle)), 0]
        )

        grid_dim = int(np.ceil(max(width_px, length_px))) + 4 * int(width_px)
        grid = np.indices((grid_dim, grid_dim))

        def gauss_value(arr):
            x, y = arr
            x = x - 2 * int(width_px)
            y = y - 2 * int(width_px)
            x_gauss = np.linalg.norm(np.cross(uvec_line, np.array([x, y, 0])))
            sigma = width_px / (2 * np.sqrt(2))  # full width at 1/e * max height
            # (2 * np.sqrt(2 * np.log(10)))# full width at 1/10th max height

            gauss_val = np.exp(-(x_gauss**2) / (2 * sigma**2))

            dist = np.abs(np.dot(uvec_line, np.array([x, y, 0])) - length_px * 0.5)
            if dist > 0.5 * length_px:
                gauss_val = 0.0

            return gauss_val if gauss_val >= 0.01 else 0.0

        # start at -2*width_px, run to 2*width_px+length_px)
        grid = np.apply_along_axis(gauss_value, 0, grid)

        nonzeros = np.argwhere(grid > 0)
        min_x, min_y = nonzeros.min(axis=0)
        max_x, max_y = nonzeros.max(axis=0)
        grid = grid[min_x:max_x, min_y:max_y]
        grid = np.pad(grid, 72)

        # rotate according to quadrant
        if angle_quadrant == 0:
            pass
        elif angle_quadrant == 1:
            grid = np.rot90(grid)
        elif angle_quadrant == 2:
            grid = np.rot90(grid, 2)
        elif angle_quadrant == 3:
            grid = np.rot90(grid, 3)

        extra_padding = 8 - np.array(grid.shape) % 8
        rand_offset = np.random.randint(0, 8, 2)
        padding = np.concatenate(
            [
                (np.floor(extra_padding / 2) + rand_offset) % 8,
                (np.ceil(extra_padding / 2) - rand_offset) % 8,
            ]
        ).astype(int)
        grid = np.pad(grid, ((padding[0], padding[2]), (padding[1], padding[3])))

        self._grid = grid
        self.xy_offset = (
            32 + np.array((padding[0], padding[1]))
        ) * self.grid_resolution
        return grid
