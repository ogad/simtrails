from typing import Protocol

import numpy as np

from simtrails.detectable import Detectable
from simtrails.imager import Imager


class ContrailDetectionAlgorithm(Protocol):
    """A protocol for contrail detection algorithms.

    Child classes must implement the __call__ method, which executes the contrail detection algorithm on the given scene and returns the resulting mask.
    They should also have a n_steps attribute, which specifies the number of steps used in the algorithm.
    """

    def __call__(
        self,
        scene: Detectable,
        imager: Imager,
        pbar=None,
        ax=None,
    ) -> np.ndarray: ...

    def simulate_features(self, imager: Imager, scene: Detectable, pbar=None):
        """
        Simulates features in the given scene using the provided imager.

        Args:
            imager (Imager): The imager used to simulate features.
            scene (Detectable): The scene in which features will be simulated.
            pbar (optional): A progress bar object to track the simulation progress.

        Returns:
            None
        """
        ...


class MannsteinCDA(ContrailDetectionAlgorithm):
    """
    MannsteinCDA is a contrail detection algorithm based on the method described in the paper by Mannstein (1999).
    
    References:
        Mannstein, H. (1999). Detection of contrails and contrail-cirrus using AVHRR data. Atmospheric Research, 51(3), 185-209.

    Attributes:
        n_steps (int): The number of steps used in the algorithm.

    Args:
        **kwargs: Additional keyword arguments to configure the algorithm.

    """

    n_steps = 4

    def __init__(self, **kwargs):
        """
        Initialize the DetectionAlgorithm class.

        Args:
            **kwargs: Additional keyword arguments, passed to the config.

        Returns:
            None
        """
        from simtrails.mannstein import config, ogad_config

        if "config" in kwargs and kwargs["config"] == "ogad":
            self.config = ogad_config | kwargs
        elif (
            "config" in kwargs and kwargs["config"] == "google"
        ) or "config" not in kwargs:
            self.config = config | kwargs
        else:
            raise ValueError("Invalid config")

    def __call__(
        self,
        scene: Detectable,
        imager: Imager,
        pbar=None,
        ax=None,
        plotfield="visualise",
        norm=None,
        **plot_kwargs,
    ) -> np.ndarray:
        """
        Perform the detection algorithm on the given scene using the provided imager.

        Args:
            scene (Detectable): The scene to perform detection on.
            imager (Imager): The imager used to simulate the observation.
            pbar (Optional): Progress bar for tracking the detection progress. Default is None, which disables progress tracking.
            ax (Optional): Matplotlib axis to visualize the detection results. Default is None, which disables visualization.

        Returns:
            np.ndarray: The mask representing the detected features in the scene.
        """

        from simtrails.mannstein import mannstein_contrail_mask
        from matplotlib.colors import TwoSlopeNorm

        features = self.simulate_features(imager, scene, pbar=pbar)

        mask = mannstein_contrail_mask(features, self.config, pbar=pbar)

        if ax:
            from simtrails.misc.plotting import outline_binary

            extent = [
                -scene.xy_offset[1],
                -scene.xy_offset[1] + scene.grid.shape[1] * scene.grid_resolution,
                -scene.xy_offset[0],
                -scene.xy_offset[0] + scene.grid.shape[0] * scene.grid_resolution,
            ]

            if plotfield == "visualise" and norm is None:
                norm = TwoSlopeNorm(vcenter=286, vmax=287, vmin=265)
            else:
                norm = norm

            plot_kwargs = {"cmap": "Blues"} | plot_kwargs
            ax.imshow(
                features[plotfield],
                norm=norm,
                extent=extent,
                **plot_kwargs,
            )
            outline_binary(mask, ax=ax, extent=extent, linewidth=0.5, color="red")

        return mask

    def simulate_features(self, imager: Imager, scene: Detectable, pbar=None):
        """
        Simulates Mannstein algorithm features for a given detectable scene.

        Args:
            imager (Imager): The imager object used for simulating observations.
            scene (Detectable): The detectable scene to simulate features for.
            pbar (optional): Progress bar object for tracking simulation progress.

        Returns:
            dict: A dictionary containing the simulated features, including visualisation data and differenced features.

        """
        from simtrails.mannstein import TDIFF, TWELVE_MICRONS

        ch13 = imager.simulate_observation(scene, 13, pbar=pbar)
        ch14 = imager.simulate_observation(scene, 14, pbar=pbar)
        ch15 = imager.simulate_observation(scene, 15, pbar=pbar)

        # Define the features, differencing them from the background to optimise for gaussian smoothing
        features = {
            "visualise": ch13,  # for visualisation purposes
            TWELVE_MICRONS: ch14 - ch14.max(),
            TDIFF: ch13 - ch15 - (ch13 - ch15).min(),
        }
        return features
