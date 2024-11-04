# coding=utf-8
#
# Based on the original implementation by Authors at Google.
# Minor modifications by Oliver Driver as described in Driver et al (2024)., September 2023
#
# Copyright 2022 The Landsat Contrails Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

TWELVE_MICRONS = "temperature_12um"
TDIFF = "brightness_temperature_difference"

config = {
    "stddev_epsilon": 0.1,
    "normalized_clip_magnitude": 2.0,
    "normalized_sum_threshold": 1,  # Tuned
    "temperature_difference_threshold": 1.33,  # Tuned
    "max_contrail_num_pixels": 1000,
    "min_contrail_num_pixels": 8,  # Tuned
    "min_contrail_pixel_length": 5,  # Tuned
    "min_linear_regression_r_score": 0.25,
    "num_line_kernels": 16,  # Tuned
    "line_kernel_size_pixels": 19,
    "line_kernel_pixels_per_sigma1": 1,  # Tuned
    "line_kernel_pixels_per_sigma_delta": 2,  # Tuned
    "lowpass_kernel_pixels_per_sigma": 3.88,  # Tuned
    "lowpass_kernel_size_pixels": 5,
    "prewitt_operator_size_pixels": 15,  # Tuned
    "prewitt_operator_smoothing_pixels": 3,
    "regional_gradient_max_scale": 2.0,
    "regional_gradient_max_offset": 1.0,  # Tuned
}

ogad_config = config | {
    "temperature_difference_threshold": 0.2,
    "normalized_sum_threshold": 1,
    "min_contrail_num_pixels": 8,
    "min_contrail_pixel_length": 8,
}

del ogad_config["regional_gradient_max_scale"]
del ogad_config["regional_gradient_max_offset"]


def lowpass(image, config):
    """Applies a gaussian lowpass kernel."""
    from scipy import signal

    # The Mannstein et al paper doesn't give enough information about this kernel
    # to reproduce it exactly, it just says "Gaussian 5x5 pixel lowpass kernel",
    # so the gaussian parameters come from config so they can get tuned
    # by blackbox optimization.
    kernel_size_pixels = config["lowpass_kernel_size_pixels"]
    pixels_per_sigma = config["lowpass_kernel_pixels_per_sigma"]
    units_per_sigma = pixels_per_sigma / kernel_size_pixels
    n1, n2 = np.meshgrid(
        np.linspace(-1, 1, kernel_size_pixels), np.linspace(-1, 1, kernel_size_pixels)
    )
    distance_from_center = np.sqrt(np.square(n1) + np.square(n2))
    gaussian = (
        1
        / (units_per_sigma * np.sqrt(2 * np.pi))
        * (np.exp(-np.square(distance_from_center / units_per_sigma) / 2))
    )
    gaussian_kernel = gaussian / gaussian.sum()
    return signal.convolve2d(image, gaussian_kernel, mode="same")


def synthesize_line_kernel(degrees, config):
    """Returns a kernel for line detection at the specified angle in `degrees`."""
    size = config["line_kernel_size_pixels"]
    radians = degrees * (np.pi / 180.0)
    n1, n2 = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))

    # d is the distance of each point from the line with slope `radians`
    # which passes through the origin.
    # The formula below for `d` is derived as follows:
    # If you rotated the line by multiplying it with
    # the standard 2D rotation matrix R = [[cos r, -sin r], [sin r, cos r]],
    # it would become an entirely vertical line (x=0).
    # If you did the same thing to all the points in the kernel field,
    # then their x-coordinate becomes the distance from the line. Then:
    # np.dot(R, [n1, n2])[0] = n1 * cos(r) - n2 * sin(r)
    d = (n1 * np.cos(radians)) - (n2 * np.sin(radians))

    pixels_per_sigma1 = config["line_kernel_pixels_per_sigma1"]
    pixels_per_sigma2 = pixels_per_sigma1 + config["line_kernel_pixels_per_sigma_delta"]
    units_per_sigma1 = pixels_per_sigma1 / config["line_kernel_size_pixels"]
    gaussian1 = (1 / (units_per_sigma1 * np.sqrt(2 * np.pi))) * (
        np.exp(-np.square(d / units_per_sigma1) / 2)
    )
    units_per_sigma2 = pixels_per_sigma2 / config["line_kernel_size_pixels"]
    gaussian2 = (1 / (units_per_sigma2 * np.sqrt(2 * np.pi))) * (
        np.exp(-np.square(d / units_per_sigma2) / 2)
    )
    difference_of_gaussians = gaussian1 - gaussian2
    return zero_sum_normalize(difference_of_gaussians)


def zero_sum_normalize(kernel):
    """Normalizes kernels that are intentionally symmetrical around 0.

    Difference of Gaussians and Prewitt operators (among others) are zero-summing,
    so we normalize the positive and negative values separately, as suggested by
    http://www.imagemagick.org/Usage/convolve/#zero-summing_normalization.

    Args:
      kernel: 2D numpy array.

    Returns:
      Zero-sum normalized kernel.
    """
    pos_sum = np.sum(np.where(kernel > 0, kernel, 0))
    neg_sum = -np.sum(np.where(kernel <= 0, kernel, 0))
    return np.where(kernel > 0, kernel / pos_sum, kernel / neg_sum)


def get_regional_gradient_mask(t_12um, t_12um_stddev, config):
    """Masks out pixels on the edges of large features.

    There's a lot of ways to implement a "large scale gradient",
    and the Mannstein et al paper only specifies the size: 15px.
    Here we use a blurred Prewitt operator because it's the simplest
    to implement at varying kernel sizes for later tuning.
    https://en.wikipedia.org/wiki/Prewitt_operator.

    Args:
      t_12um: array of 12 um temperature.
      t_12um_stddev: stddev of above array.
      config: algorithm configuration parameters (eg thresholds).

    Returns:
      boolean array of whether the pixel is part of a large scale gradient,
        and therefore is not a contrail.
    """
    from scipy import signal

    mean_kernel = np.ones(config["prewitt_operator_size_pixels"], dtype=int)
    gradient_kernel = np.concatenate(
        [
            np.ones(config["prewitt_operator_smoothing_pixels"], dtype=int),
            np.zeros(
                config["prewitt_operator_size_pixels"]
                - 2 * config["prewitt_operator_smoothing_pixels"],
                dtype=int,
            ),
            -np.ones(config["prewitt_operator_smoothing_pixels"], dtype=int),
        ]
    )
    prewitt_col = np.outer(mean_kernel, gradient_kernel)
    # Zero-sum normalize it to keep the gradient in units of Kelvin,
    # for later applying equation (5).
    prewitt_col = zero_sum_normalize(prewitt_col)
    prewitt_row = prewitt_col.T
    gradient_row = signal.convolve2d(t_12um, prewitt_row, mode="same")
    gradient_col = signal.convolve2d(t_12um, prewitt_col, mode="same")
    t_12um_regional_gradient = np.sqrt(
        np.square(gradient_row) + np.square(gradient_col)
    )
    return (
        t_12um_regional_gradient
        < (config["regional_gradient_max_scale"] * t_12um_stddev)
        + config["regional_gradient_max_offset"]
    )


def mannstein_preprocessing(features, config):
    """Returns mask identifying contrail pixels.

    Args:
      features: dict containing 11um and 12um brightness temperatures.
      config: algorithm configuration parameters (eg thresholds).
      degrees: the angle to detect contrails for.

    Returns:
      Array which is 1 where a pixel is on a contrail and 0 elsewhere.
    """
    t_12um = features[TWELVE_MICRONS]
    difference = features[TDIFF]

    t_12um_inverse = -t_12um

    t_12um_inverse_smoothed = lowpass(t_12um_inverse, config)
    difference_smoothed = lowpass(difference, config)

    t_12um_inverse_signal = t_12um_inverse - t_12um_inverse_smoothed
    difference_signal = difference - difference_smoothed

    t_12um_inverse_stddev = lowpass(np.sqrt(np.square(t_12um_inverse_signal)), config)
    difference_stddev = lowpass(np.sqrt(np.square(difference_signal)), config)

    t_12um_inverse_normalized = t_12um_inverse_signal / (
        t_12um_inverse_stddev + config["stddev_epsilon"]
    )
    difference_normalized = difference_signal / (
        difference_stddev + config["stddev_epsilon"]
    )

    t_12um_inverse_clipped = np.clip(
        t_12um_inverse_normalized,
        -config["normalized_clip_magnitude"],
        config["normalized_clip_magnitude"],
    )
    difference_clipped = np.clip(
        difference_normalized,
        -config["normalized_clip_magnitude"],
        config["normalized_clip_magnitude"],
    )

    normalized_sum = t_12um_inverse_clipped + difference_clipped

    if "regional_gradient_max_scale" in config:
        regional_gradient_mask = get_regional_gradient_mask(
            t_12um, t_12um_inverse_stddev, config
        )
    else:
        regional_gradient_mask = np.ones_like(normalized_sum, dtype=bool)

    return (
        normalized_sum,
        t_12um,
        t_12um_inverse_stddev,
        difference,
        regional_gradient_mask,
    )


def mannstein_one_angle_mask(
    config,
    degrees,
    normalized_sum,
    t_12um,
    t_12um_inverse_stddev,
    difference,
    regional_gradient_mask,
):
    from scipy import signal

    line_kernel = synthesize_line_kernel(degrees, config)
    detected_lines = signal.convolve2d(normalized_sum, line_kernel, mode="same")

    # `detected_lines` is the 'smallest' image because it has had the most
    # convolutions passed over it, and the np.nan values surrounding the
    # finite pixels has increased in size due to that, so
    # we mask off all the images to this smallest size.

    # OD: Comment above does not reflect the code: the convolutions are
    #     all in "same" mode, so the output is the same size as the input.
    normalized_sum = np.where(np.isfinite(detected_lines), normalized_sum, np.nan)
    difference = np.where(np.isfinite(detected_lines), difference, np.nan)
    regional_gradient_mask = np.where(
        np.isfinite(detected_lines), regional_gradient_mask, False
    )

    line_mask = (
        (detected_lines > config["normalized_sum_threshold"])
        & (normalized_sum > config["normalized_sum_threshold"])
        & (difference > config["temperature_difference_threshold"])
        & regional_gradient_mask
    )

    line_mask = np.where(np.isfinite(detected_lines), line_mask, np.nan)
    return line_mask


def get_angles(config):
    """Gets the list of angles from a config.

    Args:
      config: the config

    Returns:
      the angles.
    """
    return np.linspace(0, 180, config["num_line_kernels"], endpoint=False)


def label_blobs(line_mask, config):
    """Given a mask, yields pixel coordinates of connected components.

    Args:
      line_mask: 2d array, 1 when there is a contrail 0 otherwise.
      config: params for what is an allowable contrail.

    Yields:
      Tuple of (row_coordinates, col_coordinates).
    """
    from scipy.ndimage import measurements

    # measurements.label interprets np.nan as different from zero, so it thinks
    # nan's are contrail pixels. Replace all nans with zeros so we only label
    # the non-nan pixels as contrails
    line_mask = np.where(np.isnan(line_mask), 0, line_mask)
    labels, _ = measurements.label(line_mask, structure=np.ones((3, 3)))
    bincount = np.bincount(labels.flatten())
    bincount[0] = 0  # OD: ignore the background label
    (right_sized_labels,) = (
        (bincount > config["min_contrail_num_pixels"])
        & (bincount < config["max_contrail_num_pixels"])
    ).nonzero()

    for label in right_sized_labels:
        yield (labels == label).nonzero()


def linear_fit(xs, ys):
    from sklearn import linear_model

    train_ys = ys.reshape(-1, 1)
    train_xs = xs.reshape(-1, 1)
    regression = linear_model.LinearRegression().fit(train_xs, train_ys)
    return (
        regression.score(train_xs, train_ys),
        regression.coef_[0][0],
        regression.intercept_[0],
    )


def linearity_score(xs, ys, slope, intercept, treat_horizontal=False):
    """Calculate the "linearity score" of a set of points relative to a line.
    OD: Original addition to combat the problem of r-squared omitting lines that are aligned with the x or y axis.
    """
    from simtrails.misc.geometry import (
        perpendicular_distance_to_line,
        perpendicular_line,
    )

    perp_res = perpendicular_distance_to_line(xs, ys, slope, intercept)

    # early exit if the line perfectly vertical or horizontal
    if slope == 0:
        return 1 - np.sum(perp_res**2) / np.sum(np.square(xs - xs.mean()))

    if treat_horizontal:
        perp_slope, perp_intercept = perpendicular_line(slope, intercept, x=xs.mean())
    else:
        perp_slope, perp_intercept = perpendicular_line(slope, intercept, y=ys.mean())

    para_res = perpendicular_distance_to_line(xs, ys, perp_slope, perp_intercept)

    return 1 - np.sum(perp_res**2) / np.sum(para_res**2)


def linear_and_long_enough(ys, xs, config, avg_size=3):
    """Decides if a connected component is linear and long enough for a contrail.

    Does this by performing a linear fit to all the passed in pixels. Note that
    if the pixels are a vertically-oriented rectangle, the 'best fit line' will
    have a slope of infinity, but your average linear fitter really doesn't like
    making an infinite slope. So in this method we decide whether to use
    x or y as the independent variable, in such a way as to minimize the slope
    before doing any fitting (e.g. for mostly vertical clusters, we fit
    x = f(y), for mostly horizontal we fit y = f(x).

    Args:
      ys: pixel row coordinates
      xs: pixel col coordinates
      config: Dict containing params for what makes a contrail
      avg_size: Below we handle top/bottom or left/right endpoints separately,
        based on the slope. To reduce noise from extremal pixel locations, take
        the average of a small number of extremal pixels as the endpoint.

    Returns:
      True if the pixel blob meets config-defined linearity and size thresholds.
    """
    pixel_length, score = get_pixel_length(xs, ys, avg_size=avg_size)

    return (
        score > np.square(config["min_linear_regression_r_score"])
        and pixel_length > config["min_contrail_pixel_length"]
    )


def get_pixel_length(xs, ys, avg_size=3):
    sorted_indices = np.argsort(ys)
    y1 = np.mean(ys[sorted_indices][:avg_size])
    y2 = np.mean(ys[sorted_indices][-avg_size:])
    sorted_indices = np.argsort(xs)
    x1 = np.mean(xs[sorted_indices][:avg_size])
    x2 = np.mean(xs[sorted_indices][-avg_size:])

    if abs(x1 - x2) > abs(y1 - y2):
        # This is a horizontal-ish line; use x as independent variable.
        score, slope, intercept = linear_fit(xs=xs, ys=ys)
        # OD: replace r-squared with linearity score
        score = linearity_score(xs, ys, slope, intercept, treat_horizontal=True)
        y1 = x1 * slope + intercept
        y2 = x2 * slope + intercept
        pixel_length = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
    else:
        # This is a vertical-ish line; use y as independent variable.
        score, slope, intercept = linear_fit(xs=ys, ys=xs)
        # OD: replace r-squared with linearity score
        score = linearity_score(ys, xs, slope, intercept, treat_horizontal=False)
        x1 = y1 * slope + intercept
        x2 = y2 * slope + intercept
        pixel_length = np.sqrt(np.square(y1 - y2) + np.square(x1 - x2))

    return pixel_length, score


def find_contrail_pixels(line_mask, config):
    """Returns lat/lng endpoints of all contrails in line_mask.

    Args:
      line_mask: np.array, binary mask of where contrail lines were detected.
      config: dict, algorithm configuration parameters (eg thresholds).

    Returns:
      List of (row pixel coordinates, col pixel coordinates) tuples.
    """
    contrail_pixel_coords = []
    for rows, cols in label_blobs(line_mask, config):
        if not linear_and_long_enough(rows, cols, config, avg_size=1):
            continue
        contrail_pixel_coords.append((rows, cols))
    return contrail_pixel_coords


def mannstein_contrail_mask(features, config, pbar=None):
    """Mannstein et al contrail detection algorithm.

    See "Operational detection of contrails from NOAA-AVHRR-data",
    International Journal of Remote Sensing, 1999:
    https://www.tandfonline.com/doi/abs/10.1080/014311699212650

    Args:
      features: dict containing 11um-12um and 12um brightness temperatures.
      config: algorithm configuration parameters (eg thresholds).

    Returns:
      Mask, pixels identified as contrails are 1 and everywhere else is 0.
    """
    contrail_mask = np.zeros_like(features[TDIFF], dtype=int)
    args = mannstein_preprocessing(features, config)

    for degrees in get_angles(config):
        line_mask = mannstein_one_angle_mask(config, degrees, *args)
        for rows, cols in find_contrail_pixels(line_mask, config):
            contrail_mask[rows, cols] = 1

    if pbar:
        pbar.update()

    # dont allow the very edges to be contrails.... this is a bit of a hack and is needed for cirrus background
    contrail_mask[:, 0] = 0
    contrail_mask[:, -1] = 0
    contrail_mask[0, :] = 0
    contrail_mask[-1, :] = 0

    return contrail_mask
