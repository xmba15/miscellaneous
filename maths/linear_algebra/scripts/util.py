#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import open3d as o3d


def estimate_3d_bounding_box_vertices(points):
    if not isinstance(points, np.ndarray):
        print("only deal with numpy points")
        exit(0)

    if (len(points.shape) != 2 and points.shape[1] != 3):
        print("only deal with 3d data")
        exit(0)

    assert(len(points) >= 2)

    right = np.max(points[:, 0])
    left = np.min(points[:, 0])

    far = np.max(points[:, 1])
    near = np.min(points[:, 1])

    top = np.max(points[:, 2])
    bottom = np.min(points[:, 2])

    bbox_vertices = []
    for x in (left, right):
        for y in (near, far):
            for z in (bottom, top):
                bbox_vertices.append([x, y, z])

    return bbox_vertices


def setup_lineset_3dbbox(vertices):
    assert(len(vertices) == 8)

    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [[1, 0, 0] for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set
