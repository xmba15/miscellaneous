#!/usr/bin/env python
from .types import _InternalRectangle, Rectangle, get_smallest_rectangle
import numpy as np


def get_min_area_rectangle_of_hull(xy_s: np.array) -> Rectangle:
    min_rect = Rectangle()
    min_area = np.inf

    num_points = len(xy_s)

    for i in range(num_points):
        first = xy_s[i]
        second = xy_s[(i + 1) % num_points]
        axis0 = second - first
        axis0 /= np.linalg.norm(axis0)
        axis1 = np.array([-axis0[1], axis0[0]])

        min0 = np.inf
        max0 = -np.inf
        max1 = 0
        for j in range(num_points):
            axis0_projected = axis0.dot(xy_s[j] - first)
            if axis0_projected < min0:
                min0 = axis0_projected
            if axis0_projected > max0:
                max0 = axis0_projected

            axis1_projected = axis1.dot(xy_s[j] - first)
            if axis1_projected > max1:
                max1 = axis1_projected

        cur_area = (max0 - min0) * max1
        if 0 < cur_area < min_area:
            min_area = cur_area
            min_rect.center = first + axis0 * (min0 + max0) / 2 + axis1 * max1 / 2
            min_rect.axes = [axis0, axis1]
            min_rect.extents = [(max0 - min0) / 2, max1 / 2]

    return min_rect


def get_min_area_rectangle_of_hull_2(xy_s: np.array) -> Rectangle:
    min_rect = _InternalRectangle()
    min_rect.area = np.inf

    num_points = len(xy_s)

    for i in range(num_points):
        cur_rect = get_smallest_rectangle(i, (i + 1) % num_points, xy_s)
        if cur_rect.area < min_rect.area:
            min_rect = cur_rect

    rectangle = min_rect.to_rectangle(xy_s)

    return rectangle
