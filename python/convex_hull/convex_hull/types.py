#!/usr/bin/env python
from typing import Optional
import numpy as np


__all__ = ["Rectangle"]


class Rectangle:
    def __init__(
        self,
        center: Optional[np.array] = None,
        axes: Optional[list] = None,
        extents: Optional[list] = None,
    ):
        self.center = center
        self.axes = axes
        self.extents = extents

    def get_four_corners(self):
        assert self.center is not None
        assert self.axes is not None
        assert self.extents is not None

        return [
            self.center
            - self.extents[0] * self.axes[0]
            - self.extents[1] * self.axes[1],
            self.center
            + self.extents[0] * self.axes[0]
            - self.extents[1] * self.axes[1],
            self.center
            + self.extents[0] * self.axes[0]
            + self.extents[1] * self.axes[1],
            self.center
            - self.extents[0] * self.axes[0]
            + self.extents[1] * self.axes[1],
        ]


class _InternalRectangle:
    def __init__(
        self,
        axes: Optional[list] = None,
        indices: Optional[list] = None,
        area: Optional[list] = None,
    ):
        self.axes = axes
        self.indices = indices
        self.area = area

    def _project_point_to_line(
        self, out_point: np.array, in_point: np.array, direction: np.array
    ) -> np.array:
        return in_point + (out_point - in_point).dot(direction) * direction

    def _distance_point_to_line(
        self, out_point: np.array, in_point: np.array, direction: np.array
    ) -> float:
        diff = out_point - in_point
        return np.linalg.norm(diff - direction * (diff.dot(direction)))

    def to_rectangle(self, xy_s: np.array) -> Rectangle:
        rectangle = Rectangle()
        rectangle.axes = self.axes

        right_projected_u0 = self._project_point_to_line(
            xy_s[self.indices[1]], xy_s[self.indices[0]], self.axes[0]
        )
        left_projected_u0 = self._project_point_to_line(
            xy_s[self.indices[3]], xy_s[self.indices[0]], self.axes[0]
        )
        height = self._distance_point_to_line(
            xy_s[self.indices[2]], xy_s[self.indices[0]], self.axes[0]
        )

        middle_left_right = (left_projected_u0 + right_projected_u0) / 2.0
        rectangle.center = middle_left_right + self.axes[1] * height / 2.0

        rectangle.extents = [
            np.linalg.norm(right_projected_u0 - middle_left_right),
            height / 2.0,
        ]
        return rectangle


def get_smallest_rectangle(idx0: int, idx1: int, xy_s: np.array) -> _InternalRectangle:
    min_rect = _InternalRectangle()
    axis0 = xy_s[idx1] - xy_s[idx0]
    axis0 /= np.linalg.norm(axis0)
    axis1 = np.array([-axis0[1], axis0[0]])
    min_rect.axes = [axis0, axis1]
    min_rect.indices = [idx1] * 4
    support_vertices = [np.array([0.0, 0.0])] * 4

    for i in range(len(xy_s)):
        diff = xy_s[i] - xy_s[idx1]
        v = [min_rect.axes[0].dot(diff), min_rect.axes[1].dot(diff)]

        if v[0] > support_vertices[1][0] or (
            np.isclose(v[0], support_vertices[1][0]) and v[1] > support_vertices[1][1]
        ):
            min_rect.indices[1] = i
            support_vertices[1] = v

        if v[1] > support_vertices[2][1] or (
            np.isclose(v[1], support_vertices[2][1]) and v[0] < support_vertices[2][0]
        ):
            min_rect.indices[2] = i
            support_vertices[2] = v

        if v[0] < support_vertices[3][0] or (
            np.isclose(v[0], support_vertices[3][0]) and v[1] < support_vertices[3][1]
        ):
            min_rect.indices[3] = i
            support_vertices[3] = v

    min_rect.area = (
        support_vertices[1][0] - support_vertices[3][0]
    ) * support_vertices[2][1]

    return min_rect
