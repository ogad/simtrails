import numpy as np


def perpendicular_distance_to_line(x, y, slope, intercept):
    """Calculates the perpendicular distance from a point to a line.

    Args:
      x: x coordinate of point
      y: y coordinate of point
      slope: slope of line
      intercept: intercept of line

    Returns:
      Perpendicular distance from point to line.
    """
    return abs(slope * x - y + intercept) / np.sqrt(np.square(slope) + 1)


def perpendicular_line(slope, intercept, x=None, y=None):
    """Calculates the perpendicular line to a line at a point.

    Args:
      slope: slope of line
      intercept: intercept of line
      x: x coordinate of point
      y: y coordinate of point

    Returns:
      Slope and intercept of perpendicular line.
    """
    perp_slope = -1 / slope
    if y is None:
        y = slope * x + intercept
    else:
        x = (y - intercept) / slope
    perp_intercept = y - perp_slope * x
    return perp_slope, perp_intercept
