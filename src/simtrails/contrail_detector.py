import numpy as np

from simtrails.detectable import Detectable
from simtrails.detection_algorithms import (
    ContrailDetectionAlgorithm,
)
from simtrails.imager import Imager


class ContrailDetector:
    """
    Class representing a contrail detector (imager-algorithm combination).

    Attributes:
        imager (Imager): The imager used for capturing scenes.
        algorithm (ContrailDetectionAlgorithm): The algorithm used for contrail detection.

    """

    def __init__(self, imager: Imager, algorithm: ContrailDetectionAlgorithm):
        self.imager = imager
        self.algorithm = algorithm

    def contrail_mask(
        self, scene: Detectable, pbar=None, ax=None, **plot_kwargs
    ) -> np.ndarray:
        """
        Generates a contrail mask for a given detectable scene.

        Args:
            scene (Detectable): The scene to generate the contrail mask for.
            pbar (tqdm.tqdm, optional): Progress bar to display the progress of the contrail detection algorithm.
            ax (matplotlib.axes.Axes, optional): Axes to plot the scene and contrail mask.

        Returns:
            np.ndarray: The contrail mask.

        """
        from simtrails.validation import validated_contrail_mask

        if pbar is None:
            from tqdm import tqdm

            pbar = tqdm(total=self.algorithm.n_steps)

        mask = self.algorithm(scene, self.imager, pbar=pbar, ax=ax, **plot_kwargs)
        mask = validated_contrail_mask(mask, orientation=np.deg2rad(90 - scene.angle))
        return mask


class GOESMannsteinDetector(ContrailDetector):
    """
    A class representing a contrail detector using GOES imager and Mannstein algorithm.

    Attributes:
        imager (GOESImager): The GOES imager object used for contrail detection.
        algorithm (MannsteinCDA): The Mannstein algorithm object used for contrail detection.

    Args:
        resolution (float): The resolution of the imager in kilometers.
    """

    def __init__(self, resolution: float = 0.5, **mannstein_kwargs):
        from simtrails.detection_algorithms import MannsteinCDA
        from simtrails.imager import GOESImager

        self.imager = GOESImager(resolution=resolution)
        self.algorithm = MannsteinCDA(**mannstein_kwargs)
