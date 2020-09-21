#!/usr/bin/env python
import copy
from .types import _InternalRectangle, Rectangle, get_smallest_rectangle
import numpy as np
import functools


def _compute_angles(box: _InternalRectangle, xy_s: np.array):
    num_points = len(xy_s)
    angles = []

    for i in range(4):
        if box.indices[i] == box.indices[(i + 1) % 4]:
            continue
        direction = box.axes[i % 2] * (-1) ** (i > 1)
        diff = xy_s[(box.indices[i] + 1) % num_points] - xy_s[box.indices[i]]
        diff /= np.linalg.norm(diff)
        angle = np.arccos(diff.dot(direction))
        angles.append([np.rad2deg(angle), i])

    return angles


def update_box(box, angles, xy_s, visited) -> _InternalRectangle:
    num_points = len(xy_s)
    min_angle = angles[0][0]
    for angle in angles:
        if np.isclose(angle[0], min_angle):
            box.indices[angle[1]] = (box.indices[angle[1]] + 1) % num_points

    visited[box.indices[angles[0][1]]] = True

    new_box = _InternalRectangle()
    new_box.indices = [None] * 4
    for i in range(4):
        new_box.indices[i] = box.indices[(i + angles[0][1]) % 4]

    axis0 = xy_s[new_box.indices[0]] - xy_s[(new_box.indices[0] - 1) % num_points]
    axis0 /= np.linalg.norm(axis0)
    axis1 = np.array([-axis0[1], axis0[0]])
    new_box.axes = [axis0, axis1]
    diffs = [
        xy_s[new_box.indices[1]] - xy_s[new_box.indices[3]],
        xy_s[new_box.indices[2]] - xy_s[new_box.indices[0]],
    ]
    new_box.area = new_box.axes[0].dot(diffs[0]) * new_box.axes[1].dot(diffs[1])

    return new_box


def get_min_area_rectangle_of_hull_rotating_calipers(xy_s: np.array) -> Rectangle:
    num_points = len(xy_s)
    visited = [False] * num_points
    min_rect = get_smallest_rectangle(0, 1, xy_s)
    visited[1] = True

    box = copy.deepcopy(min_rect)
    for i in range(num_points):
        angles = _compute_angles(box, xy_s)

        def _compare_angles(e1, e2):
            if e1[0] < e2[0]:
                return -1
            elif e1[0] > e2[0]:
                return 1
            else:
                return 0

        angles.sort(key=functools.cmp_to_key(_compare_angles))

        if visited[(box.indices[angles[0][1]] + 1) % num_points]:
            break

        box = update_box(box, angles, xy_s, visited)

        if box.area < min_rect.area:
            min_rect = copy.deepcopy(box)

    return min_rect.to_rectangle(xy_s)
