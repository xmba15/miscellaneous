#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))
from convex_hull.convex_hull import (
    get_convex_hull_indices,
)
from convex_hull.rotating_calipers import get_smallest_rectangle


def main():
    xy_s = np.array([[5, 13], [25, 13], [25, 21], [5, 21]], dtype=float)
    plt.scatter(xy_s[:, 0], xy_s[:, 1], c="red")

    convex_hull_indices = get_convex_hull_indices(xy_s)
    assert list(convex_hull_indices) == [0, 1, 2, 3]
    _smallest_box = get_smallest_rectangle(0, 1, xy_s[convex_hull_indices])
    assert list(_smallest_box.indices) == [1, 2, 3, 0]

    smallest_box = _smallest_box.to_rectangle(xy_s)
    min_rect_corners = smallest_box.get_four_corners()
    for i in range(4):
        start = min_rect_corners[i]
        end = min_rect_corners[(i + 1) % 4]
        plt.plot([start[0], end[0]], [start[1], end[1]], marker="D", color="blue")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()
