from copy import deepcopy
import numpy as np
from simtrails.detection_algorithms import MannsteinCDA
from simtrails.imager import Imager
from simtrails.contrail import Contrail

from simtrails.contrail_detector import ContrailDetector, GOESMannsteinDetector
from simtrails.sensitivity_test import CDASensitivityTest

goes = Imager.from_name("GOES_R_ABI")
trail = Contrail(30, angle=30, resolution=0.25)


def get_imager(resolution):
    imager = deepcopy(goes)
    for ch in imager.channels.keys():
        imager.channels[ch].resolution = resolution
    return imager


def get_mask(resolution, trail=trail, ax=None):
    imager = get_imager(resolution)
    detector = ContrailDetector(imager, MannsteinCDA())
    mask = detector.contrail_mask(trail, ax=ax)
    return mask


def make_sensitivity(
    agg=np.sum,
    n_repeats=4,
    **test_vars,
):
    test = CDASensitivityTest(
        detector_generator=GOESMannsteinDetector,
        detectable_generator=Contrail,
    )
    px_detected_repeats = test.repeats(n_repeats, test_vars, n_proc=4, agg=agg)
    return px_detected_repeats
