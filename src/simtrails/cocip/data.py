# %%
from pathlib import Path
import pandas as pd
import xarray as xr
import logging
import numpy as np
from tqdm import tqdm

from simtrails.cocip.analysis import histogram
from simtrails.cocip.contrail_aggregation import aggregate_contrails
from simtrails.cocip.observability_histograms import find_observable
from simtrails.misc.microphysics import eff_radius_from_vol, iwc_kgkg1_to_gm3
from simtrails.sensitivity_result import SensitivityResult
from simtrails.validation import check_bounding
from simtrails.cocip.analysis import bin_edges_from_midpoints, is_logspace

from google.cloud import storage


class CocipDataset:
    """An object to store and manipulate datasets of CoCiP data.
    
    Attributes:
        df_waypoints (pd.DataFrame): The waypoints data.
        units_system (str): The units system of the data.
            - "preprocessed": The data is in the preprocessed units system used by simtrails.
            - "pycontrails": The data is in the pycontrails units system.
    """
    def __init__(self, df_waypoints, contrails=None, units_system="pycontrails"):

        if units_system == "pycontrails":
            df_waypoints = preprocess(df_waypoints)
            units_system = "preprocessed"
        if contrails is not None and units_system == "pycontrails":
            raise NotImplementedError(
                "Contrails are not yet preprocessed, please preprocess the contrails first"
            )

        self.df_waypoints = df_waypoints
        self.df_waypoints.drop(
            columns=[
                # "latitude",
                # "longitude",
                "sin_a",
                "cos_a",
                "air_temperature",
                "specific_humidity",
                "n_ice_per_m",
                "u_wind",
                "v_wind",
                "vertical_velocity",
                "area_eff",
                "sdr",
                "rsr",
                "olr",
            ],
            inplace=True,
        )
        self.units_system = units_system
        self._df_contrails = None if contrails is None else contrails

    @classmethod
    def from_pq(cls, time):
        """Load the dataset from a parquet file in the data store, given a timestamp."""
        return cls(open_parquet(time))

    @classmethod
    def from_gcs(cls, dt, **gcs_kwargs):
        """Get the dataset from the Google Cloud Storage bucket. If cached,
        the dataset is opened from the data store."""
        return cls(open_gcs(dt, **gcs_kwargs))

    @classmethod
    def sample(cls, fuel="jet-a"):
        """Load the sample dataset from the data store."""
        df = pd.read_parquet(
            Path(__file__).parent.parent.parent.parent
            / f"data/CoCiP/{fuel}/data_sample_preproc_{fuel}.pq"
        )
        return cls(df, units_system="preprocessed")

    @classmethod
    def join(cls, datasets):
        """Join multiple datasets into a single dataset."""
        df = pd.concat([d.df_waypoints for d in datasets])
        return CocipDataset(df, units_system="preprocessed")

    @property
    def df_contrails(self):
        if self._df_contrails is None:
            self._df_contrails = aggregate_contrails(self.df_waypoints)
        return self._df_contrails

    def sel(self, querystring, **kwargs):
        """Select a subset of the dataset based on a query string."""
        return CocipDataset(
            self.df_waypoints.query(querystring, resolvers=[kwargs]),
            units_system="preprocessed",
        )

    def histogram(
        self,
        dims,
        bins=100,
        weights=None,
        join_contrails=False,
        df_subset_function=lambda df: df,
    ):
        """Calculate a histogram of the dataset, given dimensions and bins.

        Args:
            dims (str): The dimensions to bin the data by.
            bins (int, np.ndarray, optional): The bin specification. Defaults to 100.
            weights (np.ndarray, optional): Weights to apply. Defaults to None.
            join_contrails (bool, optional): Whether to combine waypoints. Defaults to False.
            df_subset_function (Callable, optional): A function with which to filter the contrails before binning. Defaults to lambdadf:df.

        Returns:
            np.ndarray: n-dimensional histogram of the data.
            np.ndarray: The edges of the bins.
        """
        df = self.df_contrails if join_contrails else self.df_waypoints
        df = df_subset_function(df)

        if not isinstance(bins, int):
            bins = [np.array([-np.inf, *bin_list, np.inf]) for bin_list in bins]

        total = histogram(df, dims, bins=1, weights=weights)[0].squeeze()

        return histogram(df, dims, bins=bins, weights=weights)

    # def save(self, name, force_contrails=False):
    #     if self.units_system != "preprocessed":
    #         raise ValueError(
    #             "The dataset is not preprocessed, please preprocess the dataset first"
    #         )
    #     folder = data_folder / "cocip_datasets" / name
    #     folder.mkdir(parents=True, exist_ok=True)
    #     self.df_waypoints.to_parquet(folder / "waypoints.pq")
    #     if self._df_contrails is not None or force_contrails:
    #         self.df_contrails.to_parquet(folder / "contrails.pq")

    # @classmethod
    # def load(cls, name):
    #     folder = data_folder / "cocip_datasets" / name
    #     waypoints = pd.read_parquet(folder / "waypoints.pq")
    #     if (folder / "contrails.pq").exists():
    #         contrails = pd.read_parquet(folder / "contrails.pq")
    #     return cls(waypoints, contrails=contrails, units_system="preprocessed")

    def observability_dataset(
        self,
        sensitivity,
        resolution,
        join_contrails=False,
        df_subset_function=lambda df: df,
        n_repeats=4,
        n_proc=4,
        contrail_parameters={},
        force_pass_validation=False,
        bin_cocip_data=True,
        eff_width_threshold=0.5,
    ):
        """Combine the cocip data with a sensitivity, to calculate the observability
        of the simulated contrails as a function of the particular parameter space.
        
        Args:
            sensitivity (SensitivityResult | str): The sensitivity test to use.
            resolution (float): The resolution of the observability dataset.
            join_contrails (bool, optional): Whether to combine waypoints. Defaults to False.
            df_subset_function (Callable, optional): A function with which to filter the contrails before binning. Defaults to lambdadf:df.
            n_repeats (int, optional): The number of repeats to use in the sensitivity test. Defaults to 4.
            n_proc (int, optional): The number of processes to use in the sensitivity test. Defaults to 4.
            contrail_parameters (dict, optional): The parameters to use in the contrail generator. Defaults to {}.
            force_pass_validation (bool, optional): Whether to force the validation to pass. Defaults to False.
            bin_cocip_data (bool, optional): Whether to bin the cocip data. Defaults to True.
            eff_width_threshold (float, optional): The threshold for the effective width to be considered detecable. Defaults to 0.5.
        """
        from simtrails.cocip.observability_histograms import get_observability
        from simtrails.cocip.analysis import bin_edges_from_midpoints, is_logspace

        if isinstance(sensitivity, SensitivityResult):
            result = sensitivity
            if "length" not in result.data.dims:
                result.data["length"] = (
                    50  # TODO: remake saved results s.t. this is not necessary
                )
        elif isinstance(sensitivity, str):
            result = SensitivityResult.from_name(sensitivity, n_proc=n_proc)
            if "length" not in result.data.dims:
                raise ValueError("You need the length! Maybe using old saved sensitivity?")
        else:
            # do sensitivity test
            test = simtrails.CDASensitivityTest(
                detector_generator=simtrails.contrail_detector.GOESMannsteinDetector,
                detectable_generator=simtrails.sensitivity_test.InstanceGenerator(
                    simtrails.contrail.Contrail, **contrail_parameters
                ),
            )
            result = test.repeats(
                n_repeats,
                sensitivity,
                n_proc=n_proc,
            )
            result.add_area()

        # observability based on retrieved effective width
        # result.data["observability"] = (
        #     result.data["sensitivity_detections"].mean("repeat")
        #     / (result.data.length / result.data.resolution)
        # ).squeeze()

        # Observability based on probability of detection
        result.data["observability"] = (
            (
                (
                    result.data["sensitivity_detections"]
                    / (result.data.length / result.data.resolution)
                )
                > eff_width_threshold
            )
            .mean("repeat")
            .squeeze()
        )

        # get the observability
        hist_obs, dims, values = get_observability(resolution, result)
        observability = np.pad(hist_obs, 1, constant_values=np.nan)

        # get the histograms
        edges = [bin_edges_from_midpoints(v, log=is_logspace(v)) for v in values]
        coords = {d: [-np.inf, *values[i], np.inf] for i, d in enumerate(dims)}

        if bin_cocip_data:
            weights_list = [None, "rf_net", "rf_lw", "rf_sw"]

            hists_edges_totals = [  # doesn't have the totals in any more
                self.histogram(
                    dims,
                    bins=edges,
                    join_contrails=join_contrails,
                    df_subset_function=df_subset_function,
                    weights=w,
                )
                for w in weights_list
            ]
            hists = [h for (h, _) in hists_edges_totals]
            # totals = [t for _, t in hists_edges_totals]

            var_names = ["occurrence", "rf", "rf_lw", "rf_sw"]
        else:
            var_names, hists = [], []
        observability_dataset = xr.Dataset(
            {
                "observability": (dims, observability),
            }
            | {var: (dims, hist) for var, hist in zip(var_names, hists)},
            coords=coords,
        )
        # for var, total in zip(var_names, totals):
        #     observability_dataset[var].attrs["total"] = total
        observability_dataset.attrs["n_repeats"] = len(result.data["repeat"])
        check_bounding(observability_dataset, force_pass=force_pass_validation)

        return observability_dataset

    def bin_waypoints(self, binned_data: xr.Dataset | xr.DataArray):
        """Bin the waypoints to be the same as binned data.
        
        Args:
            binned_data (xr.Dataset | xr.DataArray): The data to bin the waypoints to.
        """
        # infer bins from binned_data
        dims_to_bin = [d for d in binned_data.dims if d in self.df_waypoints.columns]
        binned_data = binned_data.sortby(dims_to_bin)
        for dim in dims_to_bin:
            midpoints = binned_data[dim].values
            if not (~np.isfinite(midpoints)).sum() == 2:
                raise ValueError("Code assumes that edge bins are +-inf...")
            finite_midpoints = midpoints[np.isfinite(midpoints)]
            edges = bin_edges_from_midpoints(
                finite_midpoints, log=is_logspace(finite_midpoints)
            )
            self.df_waypoints[f"{dim}_bin_i"] = pd.cut(
                self.df_waypoints[dim],
                edges,
                labels=np.arange(len(finite_midpoints)) + 1,
            )
            self.df_waypoints[f"{dim}_bin_i"] = np.where(
                self.df_waypoints[dim] < min(edges),
                0,
                self.df_waypoints[f"{dim}_bin_i"],
            )
            self.df_waypoints[f"{dim}_bin_i"] = np.where(
                self.df_waypoints[dim] > max(edges),
                len(finite_midpoints) + 1,
                self.df_waypoints[f"{dim}_bin_i"],
            )
            self.df_waypoints[f"{dim}_bin_i"] = self.df_waypoints[
                f"{dim}_bin_i"
            ].astype(int)

    def mark_existent(self):
        """Mark the waypoints as existent if they have a time."""
        self.df_waypoints["existent"] = (~np.isnan(self.df_waypoints["time"])).astype(
            bool
        )

    def mark_observable(
        self,
        observable: xr.DataArray,
        suffix="",
    ):
        """Mark the waypoints as observable if they are within the observable data.
        
        This method adds a data variable ``observable_SUFFIX``, with values ``True`` or ``False``.
        
        Args:
            observable (xr.DataArray): The observable data (a thresholded observability dataset).
            suffix (str, optional): The suffix to add when marking the status. Defaults to "".
        """
        required_bin_cols = [f"{dim}_bin_i" for dim in observable.dims]
        if not all([col in self.df_waypoints.columns for col in required_bin_cols]):
            self.bin_waypoints(observable)

        observable = observable.sortby(list(observable.dims))

        indexer_arrays = {
            dim: xr.DataArray(self.df_waypoints[dim], dims="waypoint")
            for dim in observable.dims
        }

        self.df_waypoints["observable" + suffix] = observable.sel(
            **indexer_arrays, method="nearest"
        )

    def add_observability_status(self, observable, suffix=""):
        """Add the observability status to the waypoints.
        
        This method adds a data variable with values ``observable``, ``unobservable``, ``too optically thin``, or ``too narrow``.
        It also adds the minimum detectable width of the observability dataset.

        Args:
            observable (xr.DataArray): The observable data (a thresholded observability dataset).
            suffix (str, optional): The suffix to add when marking the status. Defaults to "".
        """
        
        self.df_waypoints["observability_status" + suffix] = np.where(
            self.df_waypoints["observable" + suffix], "observable", "unobservable"
        )
        min_width = (
            xr.where(observable, observable.width, np.inf).min("width").fillna(np.inf)
        )
        min_width = min_width.sortby(list(min_width.dims))

        indexer_arrays = {
            dim: xr.DataArray(self.df_waypoints[dim], dims="waypoint")
            for dim in min_width.dims
        }

        self.df_waypoints["min_width" + suffix] = min_width.sel(
            **indexer_arrays, method="nearest"
        )
        self.df_waypoints["observability_status" + suffix] = np.where(
            (self.df_waypoints["observability_status" + suffix] == "unobservable")
            & (~np.isfinite(self.df_waypoints["min_width" + suffix])),
            "too optically thin",
            self.df_waypoints["observability_status" + suffix],
        )

        self.df_waypoints["observability_status" + suffix] = np.where(
            (self.df_waypoints["observability_status" + suffix] == "unobservable")
            & (self.df_waypoints["min_width" + suffix]),
            "too narrow",
            self.df_waypoints["observability_status" + suffix],
        )

    def analyse_observability(
        self,
        obs_dataset: xr.Dataset,
        observability_threshold: float = 0.5,
        low_threshold: float = 0.75,
        high_threshold: float = 0.25,
        fill_from_min: bool = True,
        suffix="",
    ):
        """A pipeline to analyse the observability of the waypoints after the
        observability dataset has been calculated, including variabiiltiy in the
        observability status from threshold variability.
        
        Args:
            obs_dataset (xr.Dataset): The observability dataset.
            observability_threshold (float, optional): The threshold for observability. Defaults to 0.5.
            low_threshold (float, optional): The threshold for low observability. Defaults to 0.75.
            high_threshold (float, optional): The threshold for high observability. Defaults to 0.25.
            fill_from_min (bool, optional): Whether to fill the observability from the minimum detectable width. Defaults to True.
            suffix (str, optional): The suffix to add to the data variables. Defaults to "".
        """
        if isinstance(obs_dataset, SensitivityResult):
            raise ValueError(
                "Observability dataset must be calcuated (to ensure bins include overflows)"
            )

        obs_base, obs_low, obs_high = [
            find_observable(
                obs_dataset.observability,
                observability_threshold=obs_threshold,
                fill_from_min=fill_from_min,
            )
            for obs_threshold in [
                observability_threshold,
                low_threshold,
                high_threshold,
            ]
        ]

        self.mark_existent()
        for obs, obs_suffix in zip(
            [obs_base, obs_low, obs_high], ["", "_low", "_high"]
        ):
            self.mark_observable(obs, suffix=obs_suffix + suffix)
            self.add_observability_status(obs, suffix=obs_suffix + suffix)

    def observability_proportions(
        self,
        suffix="",
        date_range=None,
        night=True,
        day=True,
        weight_vars=[None, "rf_lw", "rf_sw", "rf_net"],
    ):
        """Get the proportion of contrails that are observable.
        
        Note:
            This function is to be executed after ``analyse_observability``.
            
        Args:
            suffix (str, optional): The suffix on the data variables to use. Defaults to "".
            date_range (tuple, optional): The date range to consider. Defaults to None.
            night (bool, optional): Whether to consider (UTC) night-time. Defaults to True.
            day (bool, optional): Whether to consider (UTC) day-time. Defaults to True.
            weight_vars (list, optional): The variables to weight by. Defaults to [None, "rf_lw", "rf_sw", "rf_net"].
        """
        subset = self.df_waypoints["existent"]
        if date_range is not None:
            subset = subset & (
                (self.df_waypoints["time"] >= date_range[0])
                & (self.df_waypoints["time"] <= date_range[1])
            )
        if not night:
            subset = (
                subset
                & (self.df_waypoints["time"].dt.hour >= np.timedelta64(6, "h"))
                & (self.df_waypoints["time"].dt.hour < np.timedelta64(18, "h"))
            )
        if not day:
            subset = subset & (
                (self.df_waypoints["time"].dt.hour < np.timedelta64(6, "h"))
                | (self.df_waypoints["time"].dt.hour >= np.timedelta64(18, "h"))
            )

        proportions_by_threshold = {}
        waypoints_to_average = self.df_waypoints[subset]  # slowish
        if "rf_net" in weight_vars:
            waypoints_to_average.loc[:, "rf_net"] = np.abs(
                waypoints_to_average["rf_net"]
            )

        grps = [w if w is not None else "existent" for w in weight_vars]
        proportions = (
            waypoints_to_average[grps + ["observability_status" + suffix]]
            .groupby("observability_status" + suffix)
            .sum()
            / waypoints_to_average[grps].sum()
        )

        for thresh in ["_low", "_high"]:
            by_weight = {}
            for i, w in enumerate(grps):
                obs_pc = (
                    waypoints_to_average[
                        waypoints_to_average["observable" + thresh + suffix]
                    ][w].sum()
                    / waypoints_to_average[w].sum()
                )
                by_weight[w] = obs_pc
            proportions_by_threshold[thresh[1:]] = pd.DataFrame(
                by_weight, index=["observable" + thresh]
            )
        # print(f"\t{w if w is not None else 'Segments'}{suffix}:")
        proportions = pd.concat(
            [
                proportions,
                proportions_by_threshold["high"],
                proportions_by_threshold["low"],
            ]
        )
        # print(proportions)
        return proportions

    def unfold_time_join(self) -> xr.Dataset:
        """"Unfold" the time join of the dataset into an xarray dataset."""
        ds = xr.Dataset.from_dataframe(self.df_waypoints)
        return ds

    @classmethod
    def from_timejoined_dataset(cls, ds):
        """Create a CoCiP dataset from a time-joined xarray dataset."""
        df = ds.to_dataframe().dropna()
        return cls(df, units_system="preprocessed")


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client(project="GCLOUD_PROJECT_NAME_GOES_HERE")

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    logging.debug(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )


data_folder = Path(__file__).parent.parent.parent.parent / "data/CoCiP"


def open_parquet(time):
    """Opens a parquet file from the data store.
    
    Data is stored in the data store (data/CoCiP) as parquet files, with the filename
    format ``YYYYMMDDTHHMM.pq``.

    Args:
        time (str,pd.Timestamp): The time fo the file to open.

    Returns:
        pd.DataFrame: The cocip waypoints data.
    """
    if isinstance(time, str):
        time = pd.Timestamp(time)
        time = time.replace(day=7, month=1, year=2019)

    df = pd.read_parquet(data_folder / f"{time:%Y%m%d}T{time:%H%M}.pq")
    return df


def preprocess(df):
    """Preprocess a cocip waypoint data file.

    Args:
        df (pd.DataFrame): The cocip waypoint data with ``pycontrails`` units.

    Returns:
        pd.DataFrame: The same data with ``preprocessed`` units
    """
    df["age"] = (df.time - df.formation_time).apply(lambda x: x.seconds / 3600)
    df["eff_radius"] = eff_radius_from_vol(df.r_ice_vol, df.rhi) * 1e6  # convert to um
    df["iwc_gm3"] = iwc_kgkg1_to_gm3(df.iwc, df.altitude / 1000)
    df["iwp"] = df.iwc_gm3 * df.depth
    df["width"] *= 1e-3  # convert to km
    df["altitude"] *= 1e-3  # convert to km
    df["depth"] *= 1e-3  # convert to km
    return df


def open_gcs(dt, fuel="jet-a", keep=False):
    """Get the file from the cache or google cloud storage if not cached.

    Args:
        dt (str, pd.Timestamp): The time to get the data for.
        fuel (str, optional): The fuel to use (e.g. saf-100, jet-a). Defaults to "jet-a".
        keep (bool, optional): Whether to add to the cache. Defaults to False.

    Returns:
        pd.DataFrame: The CoCiP waypoints data.
    """
    dt = pd.Timestamp(dt)

    # fuel = "jet-a" if not saf else "saf-100"

    filename = f"{dt:%Y%m%d}T{dt:%H%M}.pq"

    target = f"{dt.year}-{fuel}/contrails/{filename}"

    cache_location = (
        data_folder / fuel / f"{dt.year}" / f"{dt.month}" / f"{dt.day}" / filename
    )

    if cache_location.exists():
        df = pd.read_parquet(cache_location)
        return df

    print("File not found, downloading from GCS")
    if keep:
        cache_location.parent.mkdir(parents=True, exist_ok=True)
    else:
        cache_location = (
            data_folder.parent.parent / "src" / "simtrails" / "tempfiles" / filename
        )

    download_blob("BUCKET_NAME_GOES_HERE", target, cache_location)
    df = pd.read_parquet(cache_location)
    if not keep:
        cache_location.unlink()
    return df


# %%

# %%

# %%
