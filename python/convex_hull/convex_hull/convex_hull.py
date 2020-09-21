#!/usr/bin/env python
import enum
import functools
import numpy as np


__all__ = [
    "get_extreme_point_indices",
    "get_interior_point_indices",
    "get_convex_hull_indices",
]


def get_extreme_point_indices(xy_s: np.array) -> np.array:
    assert len(xy_s) > 0, "empty points"
    return np.concatenate([np.argmin(xy_s, axis=0), np.argmax(xy_s, axis=0)])


def _signed_distance_point_to_line(
    out_p: np.array, in_p1: np.array, in_p2: np.array
) -> float:
    direction = in_p2 - in_p1
    normal = np.array([-direction[1], direction[0]])
    return normal.dot(out_p - in_p1)


def _is_left_point(xy, extreme_points) -> bool:
    for i in range(4):
        start = extreme_points[i]
        end = extreme_points[i + 1] if i < 3 else extreme_points[0]
        if _signed_distance_point_to_line(xy, start, end) <= 0:
            return False
    return True


def get_interior_point_indices(xy_s: np.array, extreme_points: np.array) -> np.array:
    assert len(extreme_points) == 4
    return np.where(
        np.apply_along_axis(lambda xy: _is_left_point(xy, extreme_points), 1, xy_s)
    )[0]


def get_outerior_point_indices(xy_s: np.array, extreme_points: np.array) -> np.array:
    assert len(extreme_points) == 4
    return np.where(
        ~np.apply_along_axis(lambda xy: _is_left_point(xy, extreme_points), 1, xy_s)
    )[0]


class Orientation(enum.Enum):
    COLLINEAR = 0
    CLOCKWISE = 1
    COUNTER_CLOCKWISE = 2


def get_orientation(p: np.array, q: np.array, r: np.array) -> Orientation:
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if np.isclose(val, 0):
        return Orientation.COLLINEAR

    return Orientation.CLOCKWISE if val > 0 else Orientation.COUNTER_CLOCKWISE


def compare_orientation(p0: np.array, p1: np.array, p2: np.array) -> bool:
    def _distance(p1, p2) -> float:
        return np.linalg.norm(p2 - p1)

    orientation = get_orientation(p0, p1, p2)
    if orientation == Orientation.COLLINEAR:
        return 1 if _distance(p0, p1) >= _distance(p0, p2) else -1

    return 1 if orientation == Orientation.CLOCKWISE else -1


def graham_scan_convex_hull(xy_s: np.array) -> np.array:
    # find bottom most points by y coordinate
    # sort by smaller x if there are two points with same y value
    y_min = np.min(xy_s[:, 1])
    all_y_mins_indices = np.where(
        np.apply_along_axis(lambda xy: xy[1] == y_min, 1, xy_s)
    )
    begin_idx = all_y_mins_indices[np.argmin(xy_s[all_y_mins_indices][:, 0])][0]
    p0 = xy_s[begin_idx]

    # sort remaining points by polar angle in counterclockwise
    remaining_indices = list(range(len(xy_s)))
    remaining_indices.remove(begin_idx)

    remaining_indices.sort(
        key=functools.cmp_to_key(
            lambda idx1, idx2: compare_orientation(p0, xy_s[idx1], xy_s[idx2])
        )
    )

    # remove two or more points are collinear with p0, keep only the farthest points
    m = 1
    for i in range(1, len(remaining_indices)):
        while (
            i < len(remaining_indices) - 1
            and get_orientation(
                p0, xy_s[remaining_indices[i]], xy_s[remaining_indices[i + 1]]
            )
            == Orientation.COLLINEAR
        ):
            i += 1
        remaining_indices[m] = remaining_indices[i]
        m += 1

    remaining_indices = remaining_indices[:m]
    remaining_indices.insert(0, begin_idx)

    # convex hull can only be formed with more than 3 points
    if m < 2:
        return remaining_indices

    s = remaining_indices[:3]
    for i in range(3, m + 1):
        while len(s) > 1 and (
            get_orientation(xy_s[s[-2]], xy_s[s[-1]], xy_s[remaining_indices[i]])
            != Orientation.COUNTER_CLOCKWISE
        ):
            s.pop()
        s.append(remaining_indices[i])

    return s


def get_convex_hull_indices(xy_s: np.array) -> np.array:
    extreme_point_indices = get_extreme_point_indices(xy_s)
    outerior_point_indices = get_outerior_point_indices(
        xy_s, xy_s[extreme_point_indices]
    )

    return outerior_point_indices[graham_scan_convex_hull(xy_s[outerior_point_indices])]
