from calendar import c
from itertools import product
from pathlib import Path
from re import A
from matplotlib import pyplot as plt

from matplotlib.colors import CenteredNorm, LogNorm
from matplotlib.lines import Line2D
from simtrails.atmosphere import Atmosphere

# import xarray as xr
from simtrails.saved_sensitivities import sensitivity_values
import numpy as np
import simtrails.misc.plotting

from simtrails.forcing import get_tau
from simtrails.forcing import get_rf_schumann12 as get_rf

rf_lookup_file = Path(__file__).parent.parent / "data" / "rad_forcing_lookup.nc"


def contour_taus(
    ax,
    x_var,
    y_var,
    locator=None,
    contour_kwargs={"colors": "k", "linewidths": 0.5, "linestyles": "--"},
    **kwargs,
):
    """Draw contours of optical thickness

    Args:
        ax (matplotlib.Axes): The axes to draw the contours on.
        x_var (str): The x variable on the axes
        y_var (str): The y variable on the axes
        locator (matplotlib.locators.Locator, optional): The locator to define the frequency of contours. Defaults to None.
        contour_kwargs (dict, optional): Passed to plt.contour. Defaults to {"colors": "k", "linewidths": 0.5, "linestyles": "--"}.

    Returns:
        matplotlib.contour.QuadContourSet: The resulting contour set.
    """

    x_var = str(x_var)
    y_var = str(y_var)

    tau_params = ["iwc", "depth", "eff_radius", "iwp"]

    if x_var not in tau_params and y_var not in tau_params:
        return
    else:
        if ax.get_xscale() == "log":
            x_vals = np.logspace(
                np.log10(ax.get_xlim()[0]), np.log10(ax.get_xlim()[1]), 40
            )
            if locator is None:
                locator = plt.LogLocator(numticks=10)
        else:
            x_vals = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 40)

        if ax.get_yscale() == "log":
            y_vals = np.logspace(
                np.log10(ax.get_ylim()[0]), np.log10(ax.get_ylim()[1]), 40
            )
            if locator is None:
                locator = plt.LogLocator(numticks=10)
        else:
            y_vals = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 40)

        if locator is None:
            locator = plt.MaxNLocator(nbins=10)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)

        for x, y in product(range(len(x_vals)), range(len(y_vals))):
            kwargs[x_var] = x_vals[x]
            kwargs[y_var] = y_vals[y]
            Z[y, x] = get_tau(**kwargs)
        cs = ax.contour(
            X,
            Y,
            Z,
            # label="Optical thickness",
            locator=locator,
            **contour_kwargs,
        )
        ax.clabel(cs, inline=1, fontsize="x-small", fmt="%g")
        return cs


def plot_rf(
    ax,
    x_var,
    y_var,
    sza=0,
    f_solar=1,
    norm=CenteredNorm(0, 60),
    legend=True,
    **kwargs,
):
    tau_params = ["depth", "eff_radius", "iwc", "altitude"]
    other_params = [p for p in tau_params if p != x_var and p != y_var]

    if len(other_params) == 1 and other_params[0] not in kwargs:
        raise ValueError(f"{other_params[0]} must be supplied as a kwarg")

    def rf_getter(x, y, **kwargs):
        kwargs[x_var] = x
        kwargs[y_var] = y
        return get_rf(**kwargs)

    x_vals = sensitivity_values[x_var]
    y_vals = sensitivity_values[y_var]
    rfs = simtrails.misc.plotting.Dataset.from_getter(
        x_var,
        x_vals,
        y_var,
        y_vals,
        rf_getter,
        "rf",
        **(
            kwargs
            | {
                "solar_zenith_angle": sza,
                "f_solar": f_solar,
            }
        ),
    )

    cmap = plt.get_cmap("coolwarm")
    cmap.set_over("darkred")
    cmap.set_under("darkblue")
    axs, caxs = rfs.plot_2d(
        x_var, y_var, cmap=cmap, norm=norm, cb_kwargs={"extend": "both"}, ax=ax
    )

    contours = contour_taus(ax, x_var, y_var, **kwargs)
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    cb = caxs[0]._colorbar
    cb.set_label("Radiative forcing (W/m²)")
    # legend for contour lines
    if legend:
        ax.get_figure().legend(
            [
                Line2D([], [], linewidth=1, color="k"),
                Line2D([], [], linewidth=1, color="k", linestyle="--"),
            ],
            ["Optical thickness", "Cut"],
            loc="outside lower right",
        )


def sliced_rf_plots(
    iwc_cut=0.015, r_eff_cut=20, depth_cut=1, altitude=11, sza=0, atm="us"
):
    fig, axes = plt.subplot_mosaic(
        [[".", "thickness_iwc"], ["effradius_thickness", "effradius_iwc"]],
        figsize=(8 * 16 / 10, 8),
    )

    atmosphere = Atmosphere.from_name(atm)

    plot_rf(
        axes["thickness_iwc"],
        "iwc",
        "depth",
        eff_radius=r_eff_cut,
        altitude=altitude,
        sza=sza,
        atmosphere=atmosphere,
    )
    plot_rf(
        axes["effradius_thickness"],
        "depth",
        "eff_radius",
        iwc=iwc_cut,
        altitude=altitude,
        sza=sza,
        atmosphere=atmosphere,
    )
    plot_rf(
        axes["effradius_iwc"],
        "iwc",
        "eff_radius",
        depth=depth_cut,
        altitude=altitude,
        sza=sza,
        atmosphere=atmosphere,
    )

    axes["effradius_thickness"].axvline(depth_cut, color="k", linestyle="--")
    axes["effradius_thickness"].axhline(r_eff_cut, color="k", linestyle="--")
    axes["effradius_iwc"].axvline(iwc_cut, color="k", linestyle="--")
    axes["effradius_iwc"].axhline(r_eff_cut, color="k", linestyle="--")
    axes["thickness_iwc"].axvline(iwc_cut, color="k", linestyle="--")
    axes["thickness_iwc"].axhline(depth_cut, color="k", linestyle="--")

    return fig, axes


def colorbar(h, ax, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    d = make_axes_locatable(ax)
    cax = d.append_axes("right", size=0.1, pad=0.1, axes_class=plt.Axes)
    plt.colorbar(h, cax=cax, **kwargs)

    return cax


from abc import ABC, abstractmethod
from inspect import isgenerator
from itertools import product
from matplotlib.pylab import f
import xarray as xr
import matplotlib.pyplot as plt
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


class Dataset:
    def __init__(self, data: xr.Dataset | xr.DataArray):
        self.data = data

    @classmethod
    def from_getter(cls, x_var, x_vals, y_var, y_vals, getter, name=None, **kwargs):
        data_values = cls._apply_getter(x_vals, y_vals, getter, **kwargs)
        data = xr.DataArray(
            data_values, coords=[y_vals, x_vals], dims=[y_var, x_var], name=name
        )

        return cls(data)

    @staticmethod
    def _apply_getter(x_vals, y_vals, getter, **kwargs):
        X, Y = np.meshgrid(x_vals, y_vals)
        data = np.zeros_like(X)
        for i, j in product(range(len(x_vals)), range(len(y_vals))):
            data[j, i] = getter(
                x_vals[i],
                y_vals[j],
                **kwargs,
            )
        return data

    def plot_2d(self, x_var, y_var, **kwargs):
        plotter = DatasetPlot2D(self.data)
        return plotter.plot(x_var, y_var, **kwargs)


class DatasetPlot(ABC):
    """An abstract class for plotting datasets in a slightly convoluted way.
    
    args:
        dataset (xr.Dataset): The dataset to plot.
    """
    def __init__(self, dataset: xr.Dataset):
        self.dataset = dataset

    def plot(self, *dims, plot_var=None, **kwargs):
        """The method to make the plot. calls _plot for each point in the dataset.
        ``_plot`` should be implemented in subclasses.
        
        This takes a plotting variable, and then produces a plot for each other dimension in the dataset.

        Args:
            plot_var (str, optional): Variable to plot. Defaults to None.

        Returns:
            _type_: _description_
        """
        dim_kwargs = {k: v for k, v in kwargs.items() if k in self.dataset.dims}
        kwargs = {k: v for k, v in kwargs.items() if k not in self.dataset.dims}

        if isinstance(self.dataset, xr.Dataset):
            plot_var = plot_var if plot_var is not None else self.default_var()
            try:
                to_plot = self.dataset[plot_var].sel(**dim_kwargs)
            except KeyError:
                to_plot = self.dataset[plot_var].sel(**dim_kwargs, method="nearest")
        else:
            to_plot = self.dataset.sel(**dim_kwargs)

        remaining_dims = self.remaining_dims(dims + tuple(dim_kwargs.keys()))

        # If axis/axes are specified, use them by passing them to _plot as a generator
        axes = kwargs.pop("ax", None)
        if axes is not None:
            try:
                axes = iter(axes)
            except TypeError:
                axes = iter([axes])
        else:
            axes = None

        plot_points = list(
            product(*[self.dataset[dim].values for dim in remaining_dims])
        )
        if len(plot_points) > 1:
            logging.info(f"Plotting {len(plot_points)} plots")

        plot_returns = []
        for point in plot_points:
            fixed_vars = {dim: val for dim, val in zip(remaining_dims, point)}
            if axes is None:
                _, ax = plt.subplots()
            else:
                ax = next(axes)
            try:
                data_at_point = to_plot.sel(**fixed_vars)
            except KeyError:
                data_at_point = to_plot.sel(**fixed_vars, method="nearest")
            plot_returns.append(self._plot(*dims, data_at_point, ax=ax, **kwargs))
            self.set_title(fixed_vars, ax)

        return_val = []
        for i in range(len(plot_returns[0])):
            return_val.append([plot_return[i] for plot_return in plot_returns])

        return tuple(return_val)

    @abstractmethod
    def _plot(self, *dims, plot_data, ax, **kwargs) -> tuple[plt.Axes, ...]: ...

    def default_var(self):
        if isinstance(self.dataset, xr.DataArray):
            return self.dataset.name

        var = [i for i in self.dataset.data_vars][0]

        if len(self.dataset.data_vars) > 1:
            logging.warning("plot_var not specified, using first data_var: %s", var)
        return var

    def remaining_dims(self, used_dims):
        dims = [dim for dim in self.dataset.dims if dim not in used_dims]
        if len(dims) > 0:
            logging.warning(
                f"Plotting a non-minimal dataset, may result in many plots; remaining dims: {dims}",
            )

        return dims

    def label_axes(self, x_var, y_var, ax):
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)

    def set_title(self, fixed_vars: dict, ax):
        title = ", ".join([f"{k}: {v}" for k, v in fixed_vars.items()])
        ax.set_title(title)


class DatasetPlot2D(DatasetPlot):
    """A DatasetPlot wrapper for plt.pcolormesh"""

    def _plot(self, x_dim, y_dim, plot_data, ax, cb_kwargs={}, **kwargs):
        im = ax.pcolormesh(
            self.dataset[x_dim],
            self.dataset[y_dim],
            plot_data.transpose(y_dim, x_dim),
            **kwargs,
        )

        self.label_axes(x_dim, y_dim, ax)
        d = make_axes_locatable(ax)
        cax = d.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax, **cb_kwargs)
        cax.set_ylabel(plot_data.name)

        return (ax, cax)


def outline_binary(mapimg, extent=None, ax=None, logspaced_xs=False, **kwargs):
    """Draw a line around pixels with different values in a binary image, following the pixel edges."""
    import numpy as np
    import matplotlib.pyplot as plt

    if not mapimg.any():
        return

    if ax is None:
        ax = plt.gca()

    if extent is None:
        try:
            extent = ax.images[0].get_extent()
        except IndexError:
            extent = (-0.5, mapimg.shape[1] - 0.5, -0.5, mapimg.shape[0] - 0.5)
    x0, x1, y0, y1 = extent

    # a vertical line segment is needed, when the pixels next to each other horizontally
    #   belong to diffferent groups (one is part of the mask, the other isn't)
    # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates
    ver_seg = np.where(mapimg[:, 1:] != mapimg[:, :-1])

    # the same is repeated for horizontal segments
    hor_seg = np.where(mapimg[1:, :] != mapimg[:-1, :])

    # if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
    #   (2,7) and (2,8), i.e. from (2,8)..(3,8)
    # in order to draw a discountinuous line, we add Nones in between segments
    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0] + 1))
        l.append((p[1] + 1, p[0] + 1))
        l.append((np.nan, np.nan))

    # and the same for vertical segments
    for p in zip(*ver_seg):
        l.append((p[1] + 1, p[0]))
        l.append((p[1] + 1, p[0] + 1))
        l.append((np.nan, np.nan))

    # now we transform the list into a numpy array of Nx2 shape
    segments = np.array(l)

    # now we need to know something about the image which is shown
    #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
    #   drawn with origin='lower'
    # with this information we can rescale our points
    if logspaced_xs:
        segments[:, 0] = 10 ** (
            np.log10(x0)
            + (np.log10(x1) - np.log10(x0)) * segments[:, 0] / mapimg.shape[1]
        )
    else:
        segments[:, 0] = x0 + (x1 - x0) * segments[:, 0] / mapimg.shape[1]

    segments[:, 1] = y0 + (y1 - y0) * segments[:, 1] / mapimg.shape[0]

    # and now there isn't anything else to do than plot it
    if "color" not in kwargs:
        kwargs["color"] = (1, 0, 0, 0.5)
    if "linewidth" not in kwargs and "lw" not in kwargs:
        kwargs["linewidth"] = 3
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.5
    ax.plot(segments[:, 0], segments[:, 1], **kwargs)
