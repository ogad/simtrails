from itertools import product
from typing import Any, Callable, Iterable
import numpy as np
import logging
from tqdm import tqdm
import xarray as xr

from simtrails.contrail_detector import ContrailDetector
from simtrails.detectable import Detectable
from simtrails.sensitivity_result import SensitivityResult


class InstanceGenerator:
    """
    A class that generates instances based on a given generator class and parameters.

    Args:
        generator_class (callable): The generator class or function used to generate instances.
        **kwargs: Additional keyword arguments to be passed to the generator class.

    Raises:
        ValueError: If the generator_class is not callable but kwargs were passed.

    Returns:
        The generated instance.

    """

    def __init__(self, generator_class, **kwargs) -> None:
        self.generator_class = generator_class
        self.base_kwargs = kwargs

        if not callable(self.generator_class) and kwargs:
            raise ValueError(
                f"InstanceGenerator {self.generator_class} is not callable, but kwargs were passed."
            )

    def __call__(self, **kwargs):
        """
        Generate an instance based on the given parameters.

        Args:
            **kwargs: Additional keyword arguments to be passed to the generator class.

        Raises:
            ValueError: If the generator_class is not callable but kwargs were passed.

        Returns:
            The generated instance.

        """
        if not callable(self.generator_class):
            if kwargs:
                raise ValueError(
                    f"InstanceGenerator {self.generator_class} is not callable, but kwargs were passed."
                )
            return self.generator_class
        return self.generator_class(**{**self.base_kwargs, **kwargs})

    def get_variations(self, values: dict[str, Iterable], **kwargs):
        """
        Generate variations of instances based on the given parameter values.

        Args:
            values (dict[str, Iterable]): A dictionary of parameter names and their corresponding values.

        Returns:
            variations_kwargs (list[dict]): A list of dictionaries containing the variations of parameter values.
            variations (list): A list of generated instances with different parameter values.

        """
        if values == {}:
            return {}, [self()]

        parameters = sorted(list(values.keys()))
        parameters_coords = product(*[values[parameter] for parameter in parameters])
        parameters_coords = list(zip(*parameters_coords))

        n_scenes = len(parameters_coords[0])

        variations_kwargs = [
            dict(
                zip(
                    parameters,
                    [parameter_coords[i] for parameter_coords in parameters_coords],
                )
            )
            for i in range(n_scenes)
        ]
        variations = [
            self(**variation_kwargs, **kwargs)
            for variation_kwargs in tqdm(variations_kwargs)
        ]

        return variations_kwargs, variations


class CDASensitivityTest:
    """
    Class representing a sensitivity test for contrail detection.

    Args:
        detector_generator (InstanceGenerator | Callable[..., ContrailDetector] | ContrailDetector):
            Generator or callable that produces instances of ContrailDetector.
        detectable_generator (InstanceGenerator | Callable[..., Detectable] | Detectable):
            Generator or callable that produces instances of Detectable.
    """

    def __init__(
        self,
        detector_generator: (
            InstanceGenerator | Callable[..., ContrailDetector] | ContrailDetector
        ),
        detectable_generator: (
            InstanceGenerator | Callable[..., Detectable] | Detectable
        ),
        cda_properties: dict[str, Any] = {},
    ):
        self.detector_generator = (
            detector_generator
            if isinstance(detector_generator, InstanceGenerator)
            else InstanceGenerator(detector_generator, **cda_properties)
        )
        self.detectable_generator = (
            detectable_generator
            if isinstance(detectable_generator, InstanceGenerator)
            else InstanceGenerator(detectable_generator)
        )
        # self.parameter = parameter

    @staticmethod
    def _contrail_mask(
        detector: ContrailDetector, scene: Detectable, pbar=False
    ) -> np.ndarray:
        return detector.contrail_mask(scene, pbar=pbar)

    @staticmethod
    def _contrail_agg(
        detector: ContrailDetector,
        scene: Detectable,
        agg: Callable,
        repeats: int = 1,
        pbar=False,
    ) -> np.ndarray:
        return_vals = [
            agg(detector.contrail_mask(scene, pbar=pbar)) for _ in range(repeats)
        ]
        scene.clear_memory()
        return tuple(return_vals)

    @staticmethod
    def _optical_depth(
        detector: ContrailDetector, scene: Detectable, pbar=None
    ) -> np.ndarray:
        logging.debug("Calculating an optical depth...")
        return scene.max_optical_depth(detector.imager, 13)

    @staticmethod
    def _run_sensitivity_detection(
        scenes: Iterable[Detectable],
        detectors: Iterable[ContrailDetector],
        agg=np.sum,
        n_repeats=1,
        pbar=None,
        record_optical_depth=False,  # Think this is deactivated
        n_proc=1,
    ):
        from pathos.pools import ProcessPool

        outputs = []
        optical_depths = []

        inherit_pbar = True
        if pbar is None:
            len_pbar = detectors[0].algorithm.n_steps * len(scenes) * len(detectors)
            pbar = tqdm(total=len_pbar)
            inherit_pbar = False

        logging.info("Creating product of detectors and scenes....")
        points = list(product(detectors, scenes, [agg], [n_repeats]))
        del detectors, scenes

        logging.info("Starting multiprocessing....")
        p = ProcessPool(n_proc)
        from tqdm import tqdm

        outputs = []
        with tqdm(total=len(points)) as pbar:
            for output in p.imap(_star_contrail_agg, points):
                outputs += [output]
                pbar.update(1)
        if record_optical_depth:
            from tqdm import tqdm

            optical_depths = list(
                tqdm(
                    p.imap(_star_optical_depth, points),
                    total=len(points),
                )
            )
        p.close()
        p.join()
        p.clear()
        logging.debug("Finished multiprocessing.")

        if not inherit_pbar:
            pbar.close()

        if (
            record_optical_depth
        ):  # Think this won't feed in well to repeats any more i.e. is deactivated
            return outputs, optical_depths

        return outputs, None

    def __call__(self, values: dict[str, Iterable], agg=np.sum, pbar=None) -> SensitivityResult:
        """
        Perform the sensitivity test on the given values.

        Args:
            values (dict[str, Iterable]): A dictionary of input values.
            agg (callable, optional): The aggregation function to apply. Defaults to np.sum.
            pbar (object, optional): A progress bar object. Defaults to None.

        Returns:
            Any: The result of the sensitivity test.
        """
        return self.repeats(1, values, agg)

    def repeats(
        self,
        n: int,
        values: dict[str, Iterable],
        agg=np.sum,
        n_proc=4,
        record_optical_depth=False,  # Think this is deactivated
        **kw_attrs,
    ) -> SensitivityResult:
        """
        Run multiple repeats of the sensitivity test.

        Args:
            n (int): Number of repeats.
            values (dict[str, Iterable]): Dictionary of parameter values to vary.
            agg (callable, optional): Aggregation function to apply to the results. Defaults to np.sum.
            n_proc (int, optional): Number of processes to use for parallel execution. Defaults to 4.
            record_optical_depth (bool, optional): Whether to record optical depth. Defaults to False.
            **kw_attrs: Additional keyword arguments.

        Returns:
            SensitivityResult: Result of the sensitivity test.
        """
        from simtrails.sensitivity_result import SensitivityResult

        logging.info(
            f"Running {n} repeats of sensitivity test with {n_proc} processes."
        )

        if values == {}:
            raise ValueError("No values passed to sensitivity test.")
        # if "resolution" not in values.keys():
        #     values["resolution"] = np.array([0.5])

        detector_parameters = ["resolution", "noise_equiv_temp_diff", "nedt"]

        scene_values = {
            key: val for key, val in values.items() if key not in detector_parameters
        }

        detector_values = {
            key: val for key, val in values.items() if key not in scene_values.keys()
        }

        # If there are too many scenes, should  split into chunks...
        n_scenes = np.prod(
            [
                len(scene_values[key])
                for key in scene_values.keys()
                if key not in detector_parameters
            ]
        )
        logging.info(f"Running {n_scenes} scenes.")

        variations_coords, varied_scenes = self.detectable_generator.get_variations(
            scene_values
        )
        imager_coords, varied_detectors = self.detector_generator.get_variations(
            detector_values
        )

        ordered_parameters = sorted(list(detector_values.keys())) + sorted(
            list(scene_values.keys())
        )

        if "resolution" not in values.keys():
            values["resolution"] = np.array(
                [next(iter(varied_detectors[0].imager.channels.values())).resolution]
            )
            ordered_parameters = ["resolution"] + ordered_parameters
        if "length" not in values.keys():
            values["length"] = np.array([varied_scenes[0].length])
            ordered_parameters = ["length"] + ordered_parameters

        logging.info(
            f"Running {len(varied_detectors)} detectors over {len(varied_scenes)} scenes."
        )

        results = self._run_sensitivity_detection(
            varied_scenes,
            varied_detectors,
            agg,
            n_repeats=n,
            pbar=False,
            record_optical_depth=record_optical_depth,
            n_proc=n_proc,
        )

        return SensitivityResult.from_multiprocessed_repeats(
            ordered_parameters, values, n, *results, **kw_attrs
        )


def single_contrail_length(mask):
    from .mannstein import get_pixel_length
    from scipy.ndimage import label

    labels, _ = label(mask, structure=np.ones((3, 3)))
    if mask.max() == 0:
        return 0

    if mask.max() > 1:
        print(
            "WARNING: multiple contrails detected in single_contrail_length. Measure first contrail only."
        )

    xs, ys = (labels == 1).nonzero()

    length, _ = get_pixel_length(xs, ys, avg_size=1)
    return length


def _star_contrail_agg(x):
    return CDASensitivityTest._contrail_agg(*x)


def _star_optical_depth(x):
    return CDASensitivityTest._optical_depth(*x)
