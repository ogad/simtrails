import logging
import numpy as np
import xarray as xr
from pathlib import Path


class SensitivityResult:
    """
    A class representing the results of a sensitivity analysis.

    Attributes:
        data (xr.Dataset): The dataset containing the sensitivity analysis results.
            Constructed by passing args and kwargs to xr.Dataset.
    """

    def __init__(self, *args, **kwargs):
        self.data = xr.Dataset(*args, **kwargs)

    @classmethod
    def from_name(cls, name, n_proc=4):
        """
        Create a SensitivityResult object from a given name.

        Parameters:
            name (str): The name of the dataset.

        Returns:
            cls: An instance of the sensitivity result.

        """
        from simtrails.sensitivity_test import CDASensitivityTest, InstanceGenerator
        from simtrails.contrail import Contrail, GaussianContrail
        from simtrails.contrail_detector import GOESMannsteinDetector
        from simtrails.saved_sensitivities import saved_sensitivities

        if name.split("_")[-1][0] == "v":
            version = int(name.split("_")[-1][1:])
            name = "_".join(name.split("_")[:-1])
        else:
            version = saved_sensitivities[name]["version"]

        properties = saved_sensitivities[name]["properties"]
        contrail_parameters = saved_sensitivities[name].get("contrail_properties", {})
        n_repeats = saved_sensitivities[name].get("repeats", 4)
        detectable_class = saved_sensitivities[name].get("class", "Contrail")
        cda_properties = saved_sensitivities[name].get("cda_properties", {})

        filename = f"{name}_v{version}.nc"
        filepath = Path(__file__).parent / "data" / "sensitivity_results" / filename
        if not filepath.exists():
            result = cls.from_properties(
                properties,
                n_repeats,
                detectable_class,
                n_proc,
                cda_properties=cda_properties,
                **contrail_parameters,
            )
            result.save(filepath)

        return cls.load(filepath)

    @classmethod
    def from_properties(
        cls,
        properties,
        n_repeats=4,
        detectable_class="GaussianContrail",
        n_proc=4,
        cda_properties={},
        **contrail_propeties,
    ):
        """
        Create a SensitivityResult object from a given set of properties by running a sensitivity test.

        Parameters:
            properties (dict): The properties of the sensitivity test.
            n_repeats (int): The number of repeats for the sensitivity test.
            detector_class (class): The class

        Returns:
            cls: An instance of the sensitivity result.
        """
        from simtrails.contrail import Contrail, GaussianContrail
        from simtrails.contrail_detector import GOESMannsteinDetector
        from simtrails.sensitivity_test import CDASensitivityTest, InstanceGenerator

        if detectable_class == "Contrail":
            detectable_class = Contrail
        elif detectable_class == "GaussianContrail":
            detectable_class = GaussianContrail
        else:
            raise ValueError(f"Unknown detectable class {detectable_class}")

        test = CDASensitivityTest(
            detector_generator=GOESMannsteinDetector,
            detectable_generator=InstanceGenerator(
                detectable_class, **contrail_propeties
            ),
            cda_properties=cda_properties,
        )
        result = test.repeats(n_repeats, properties, n_proc=n_proc)
        result.add_area()
        return result

    # TODO: this is broken, needs reworking for muliple optical depth drivers
    def swap_dims_optical_depth(self):
        """Deprecated"""
        print(self.data["optical_depth"])
        if "optical_depth" not in self.data.variables.keys():
            raise ValueError("Cannot swap dimensions if optical depth is not present.")
        self.data = self.data.swap_dims({"iwc": "optical_depth"})

    def plot_series(
        self,
        var1: str,  # Variable | str,
        var2: str = "resolution",  # Variable | str = Variable.RESOLUTION,
        var3: str = None,  # Variable | str = None,
        y_var: str = "area",  # Variable.AREA,
        ax=None,
        **kwargs,
    ):
        """
        Plot series of data based on specified variables.

        Parameters:
            var1 (Variable or str): The x-axis variable to plot.
            var2 (Variable or str, optional): The hue variable to plot. Defaults to Variable.RESOLUTION.
            var3 (Variable or str, optional): The multi-axis variable. Defaults to None.
            y_var (Variable, optional): The variable to plot on the y-axis. Defaults to Variable.AREA.
            ax (matplotlib.axes.Axes or list of Axes, optional): The axes to plot on. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the plotting functions.

        Raises:
            ValueError: If the number of axes does not equal the number of resultant plots.

        Returns:
            None
        """

        import seaborn as sns
        import matplotlib.pyplot as plt

        # if isinstance(var1, str):
        #     var1 = Variable[var1.upper()]
        # if isinstance(y_var, str):
        #     y_var = Variable[y_var.upper()]
        # if isinstance(var2, str):
        #     var2 = Variable[var2.upper()]
        # if isinstance(var3, str):
        #     var3 = Variable[var3.upper()]

        if (
            var1 == "optical_depth" or y_var == "optical_depth"
        ):  # Variable.OPTICAL_DEPTH or y_var == Variable.OPTICAL_DEPTH:
            self.swap_dims_optical_depth()

        try:
            ds_to_plot = self.data.sel(**kwargs)
        except KeyError:
            ds_to_plot = self.data.sel(**kwargs, method="nearest")

        if var3 is not None:
            plotting_datasets = {
                val: ds_to_plot.sel(**{str(var3): val})
                for val in ds_to_plot[str(var3)].values
            }
        else:
            plotting_datasets = {None: ds_to_plot}

        if ax is None:
            figs = [plt.figure() for _ in range(len(plotting_datasets))]
            axes = [fig.subplots() for fig in figs]
        elif isinstance(ax, plt.Axes):
            axes = [ax]
        else:
            axes = ax

        if len(axes) != len(plotting_datasets):
            raise ValueError(
                f"Number of axes ({len(axes)}) must equal number of resultant plots ({len(plotting_datasets)})"
            )
        for (title, to_plot), ax in zip(plotting_datasets.items(), axes):
            g = sns.lineplot(
                y=str(y_var),
                x=str(var1),
                hue=str(var2),
                data=to_plot.to_dataframe().reset_index(),
                ax=ax,
                # **kwargs,
            )
            sns.scatterplot(
                x=str(var1),
                y=str(y_var),
                hue=str(var2),
                data=to_plot.to_dataframe().reset_index(),
                s=5,
                alpha=0.5,
                ax=ax,
                legend=False,
            )
            # plt.xlabel(var1.label)
            # plt.ylabel(y_var.label)
            # g.get_legend().set_title(var2.label)
            # if title is not None:
            #     ax.set_title(f"{var3.label} = {title}")

    def plot_2d(
        self,
        var1: str,  # Variable | str,
        var2: str,  # Variable | str,
        hue_var: str = "area",  # Variable | str = Variable.AREA,
        cmap="Reds",
        ax=None,
        tau_locator=None,
        **kwargs,
    ):
        """
        Plot a 2D heatmap of the specified variables.

        Parameters:
            var1 (Variable or str): The x-axis variable to plot.
            var2 (Variable or str): The y-axis variable to plot.
            hue_var (Variable or str, optional): The variable to use for coloring the heatmap. Defaults to Variable.AREA.
            cmap (str, optional): The colormap to use for the heatmap. Defaults to "Reds".
            ax (matplotlib.axes.Axes, optional): The axes on which to plot the heatmap. If not provided, a new figure will be created.
            **kwargs: Additional keyword arguments to customize the plot.

        Returns:
            heatmaps (List[matplotlib.axes.Axes]): The axes objects containing the heatmaps.
            colorbars (List[matplotlib.colorbar.Colorbar]): The colorbar objects associated with the heatmaps.
        """
        # var1, var2, hue_var = [
        #     var if isinstance(var, Variable) or var is None else Variable[var.upper()]
        #     for var in [var1, var2, hue_var]
        # ]
        from simtrails.misc.plotting import contour_taus, DatasetPlot2D

        tau_params = [
            "depth",
            "eff_radius",
            "iwc",
        ]  # TODO: move to the same place as the other tau_params
        tau_kwargs = {k: v for k, v in kwargs.items() if k in tau_params}
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in tau_kwargs.keys() or k in self.data.variables.keys()
        }

        heatmaps, colorbars = DatasetPlot2D(self.data.mean("repeat")).plot(
            var1,
            var2,
            plot_var=hue_var,
            cmap=cmap,
            ax=ax,
            **kwargs,  # need to take out kwargs passed only for contour_taus
        )

        for ax in heatmaps:
            ax.set_xlabel(var1)  # .label)
            ax.set_ylabel(var2)  # .label)

            tau_params = ["depth", "eff_radius", "iwc", "iwp"]
            tau_params = [
                p
                for p in tau_params
                if p != var1 and p != var2
            ]

            if tau_locator is not None:
                tau_kwargs["locator"] = tau_locator
            contour_taus(
                ax,
                var1,
                var2,
                **tau_kwargs,
            )

        for cax in colorbars:
            cax.set_ylabel(hue_var)  # .label)

        return heatmaps, colorbars

    def save(self, path):
        """
        Save the data to a NetCDF file.

        Parameters:
            path (str): The path to save the NetCDF file.

        Returns:
            None
        """
        self.data.to_netcdf(path)

    @classmethod
    def load(cls, path):
        """
        Load a sensitivity result from a given path.

        Parameters:
            path (str): The path to the dataset.

        Returns:
            cls: An instance of the sensitivity result.
        """
        from simtrails.lookup import RENAME_DICT

        data = xr.load_dataset(path)
        data_rename_dict = {k: v for k, v in RENAME_DICT.items() if k in data}
        data = data.rename(data_rename_dict)

        return cls(data)

    @classmethod
    def from_multiprocessed_repeats(
        cls,
        ordered_parameters,
        values,
        n_repeats,
        repeat_detections,
        repeat_optical_depths=None,
        **kw_attrs,
    ):
        """
        Create a SensitivityResult object from multiprocessed repeat results.

        Args:
            cls (class): The class of the SensitivityResult object.
            multiprocessed_repeat_results (list): List of repeat results from multiprocessing.
            ordered_parameters (list): List of ordered parameters.
            values (dict): Dictionary of parameter values.
            kw_attrs: Additional keyword arguments for the SensitivityResult object.

        Returns:
            SensitivityResult: The created SensitivityResult object.
        """
        import xarray as xr
        from datetime import date

        repeat_detections = np.array(repeat_detections)
        repeat_detections = repeat_detections.reshape(-1, n_repeats).T
        if repeat_optical_depths is not None:
            repeat_optical_depths = np.array(repeat_optical_depths)
            repeat_optical_depths = repeat_optical_depths.reshape(-1, n_repeats).T

        shape = [n_repeats] + [
            len(values[parameter]) for parameter in ordered_parameters
        ]
        repeat_detections = repeat_detections.reshape(shape)
        if repeat_optical_depths is not None:
            repeat_optical_depths = repeat_optical_depths.reshape(shape)

        results = {
            "sensitivity_detections": repeat_detections,
        }
        if repeat_optical_depths is not None:
            results["optical_depth"] = repeat_optical_depths
        results = {
            key: xr.DataArray(
                results[key],
                dims=("repeat", *ordered_parameters),
                coords={"repeat": range(len(repeat_detections)), **values},
                name=key,
                attrs=dict(date=date.today().strftime("%Y-%m-%d"), **kw_attrs),
            )
            for key in results.keys()
        }

        if (
            "iwc" in results.keys()
        ) and repeat_optical_depths is not None:  # FIXME: optical depth is not a simple 1D function of IWC
            index = {key: 0 for key in results.keys() if key != "iwc"}
            results["optical_depth"] = results["optical_depth"].isel(**index, drop=True)

        return cls(results)

    @classmethod
    def from_array_repeats(cls, array_output_folder, save=True):
        """
        Load and combine multiple repeated result files into a single SensitivityResult object.

        Parameters:
            cls: The class object.
            array_output_folder: The folder path where the repeated result files are located.
            save: A boolean indicating whether to save the combined result as a new file. Default is True.

        Returns:
            results: A SensitivityResult object containing the combined results.

        Note:
            - The repeated result files should be in NetCDF format.
            - The repeated result files should have names starting with ``repeated_result_``.
            - The combined result will have expanded dimensions based on the dimensions present in the repeated result files.
            - If save is True, the combined result will be saved as "result.nc" in the array_output_folder.
        """
        import xarray as xr
        from datetime import date
        from pathlib import Path
        from glob import glob

        array_output_folder = Path(array_output_folder)

        repeat_result_files = glob("repeated_result_*.nc", root_dir=array_output_folder)
        repeat_results = [
            cls.load(array_output_folder / file).data for file in repeat_result_files
        ]
        dims_to_expand = [
            d
            for d in repeat_results[0].coords.keys()
            if d not in repeat_results[0].dims
        ]
        repeat_results = [
            repeat_result.expand_dims(dims_to_expand)
            for repeat_result in repeat_results
        ]
        results = cls(
            xr.combine_by_coords(repeat_results, combine_attrs="drop_conflicts")
        )
        if save:
            results.save(array_output_folder / "result.nc")
        return results

    def averaged_repeats(self):
        """
        Calculate the average of the data over repeated trials.

        Returns:
            The mean value of the data over the "repeat" dimension.
        """
        return self.data.mean("repeat")

    def add_area(self):
        """
        Calculates and adds the area to the data dictionary based on the sensitivity detections and resolution.
        """
        self.data["area"] = (
            self.data["sensitivity_detections"] * self.data.resolution**2
        )

    def add_percent_area(self):
        """
        Calculates and adds the percentage area to the data.

        The percentage area is calculated by dividing the area of each data point
        by the product of its length and width.
        """
        self.data["percent_area"] = self.data["area"] / (
            self.data["length"] * self.data["width"]
        )

    def add_eff_width(self):
        """
        Adds the effective width in pixels to the data dictionary.
        """
        expected_px = self.data.length / self.data.resolution
        self.data["observability"] = self.data["sensitivity_detections"] / expected_px
        self.data["eff_width"] = self.data["observability"] * self.data.resolution
