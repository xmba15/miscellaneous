#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))
from convex_hull.convex_hull import (
    get_extreme_point_indices,
    get_interior_point_indices,
    get_convex_hull_indices,
)
from convex_hull.minimum_rectangle_exhaustive_search import (
    get_min_area_rectangle_of_hull,
    get_min_area_rectangle_of_hull_2,
)
from convex_hull.rotating_calipers import (
    get_min_area_rectangle_of_hull_rotating_calipers,
)


def get_args():
    import argparse

    parser = argparse.ArgumentParser("s")
    parser.add_argument("--num_points", type=int, default=200)
    parser.add_argument("--max_x", type=float, default=60)
    parser.add_argument("--max_y", type=float, default=10)

    return parser.parse_args()


def prepare_data(num_points: int, max_x: int, max_y: int, seed: int = 2021) -> np.array:
    assert num_points > 0, "number of points needs to be > 0"
    assert max_x > 0, "max x needs to be > 0"
    assert max_y > 0, "max y needs to be > 0"

    np.random.seed(seed)
    x_s = np.random.rand(num_points) * 2 * max_x - max_x
    y_s = np.random.rand(num_points) * 2 * max_y - max_y
    return np.concatenate([x_s[:, np.newaxis], y_s[:, np.newaxis]], axis=1)


def main():
    args = get_args()
    xy_s = prepare_data(args.num_points, args.max_x, args.max_y)
    plt.scatter(xy_s[:, 0], xy_s[:, 1], c="red")

    xy_extreme_points = xy_s[get_extreme_point_indices(xy_s)]

    for i in range(4):
        start = xy_extreme_points[i]
        end = xy_extreme_points[i + 1] if i < 3 else xy_extreme_points[0]
        plt.plot([start[0], end[0]], [start[1], end[1]], marker="D", color="orange")

    interiors = xy_s[get_interior_point_indices(xy_s, xy_extreme_points)]
    plt.scatter(interiors[:, 0], interiors[:, 1], c="blue")

    convex_hull_indices = get_convex_hull_indices(xy_s)
    convex_hull = xy_s[convex_hull_indices]
    for i in range(len(convex_hull)):
        start = convex_hull[i]
        end = convex_hull[(i + 1) % len(convex_hull)]
        plt.plot([start[0], end[0]], [start[1], end[1]], marker="o", c="green")

    # exhaustive_search_min_rect = get_min_area_rectangle_of_hull_2(
    #     xy_s[convex_hull_indices]
    # )
    exhaustive_search_min_rect = get_min_area_rectangle_of_hull_rotating_calipers(
        xy_s[convex_hull_indices]
    )
    min_rect_corners = exhaustive_search_min_rect.get_four_corners()

    for i in range(4):
        start = min_rect_corners[i]
        end = min_rect_corners[(i + 1) % 4]
        plt.plot([start[0], end[0]], [start[1], end[1]], marker="D", color="blue")

    xy_conv = xy_s[convex_hull_indices]

    plt.plot([xy_conv[11][0], xy_conv[10][0]], [xy_conv[11][1], xy_conv[10][1]], marker="D", color="brown")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()
