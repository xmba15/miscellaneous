#!/usr/bin/env python
import argparse
import cv2
import numpy as np
from new_types import CameraMatrix
from direct_pose_estimation import (
    calc_direct_pose_estimation_single_layer,
    calc_direct_pose_estimation_multi_layer,
)
from liegroups.numpy import SE3


def _get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--first_image", type=str, required=True)
    parser.add_argument("--second_image", type=str, required=True)
    parser.add_argument("--disparity_image", type=str, required=True)

    return parser.parse_args()


def _gen_ref_points(
    width: int,
    height: int,
    num_points: int = 2000,
    border_margin: int = 20,
    seed: int = 2021,
):
    np.random.seed(seed)
    xs = np.random.randint(border_margin, width - 1 - border_margin, num_points)
    ys = np.random.randint(border_margin, height - 1 - border_margin, num_points)
    return cv2.hconcat((xs, ys))


def main():
    args = _get_args()
    image_paths = [args.first_image, args.second_image]
    images = [None] * 2
    gray_images = [None] * 2
    for i in range(2):
        images[i] = cv2.imread(image_paths[i])
        assert images[i] is not None, "failed to load {}".format(image_paths[i])
        gray_images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

    disparity_image = cv2.imread(args.disparity_image, 0)

    K = CameraMatrix(718.856, 718.856, 607.1928, 185.2157, 0.573)
    height, width = images[0].shape[:2]
    num_points = 2000
    border_margin = 20
    ref_points = _gen_ref_points(width, height, num_points, border_margin)
    ref_points_depth = np.zeros(num_points)
    for i, ref_point in enumerate(ref_points):
        ref_points_depth[i] = (
            K.fx * K.base_line / disparity_image[ref_point[1], ref_point[0]]
        )

    T21_SE3 = SE3.identity()
    T21_SE3, projected = calc_direct_pose_estimation_multi_layer(
        gray_images[0],
        gray_images[1],
        ref_points,
        ref_points_depth,
        K,
        T21_SE3,
        half_window_size=1,
        num_scale=4,
    )

    for ref_point, cur_point in zip(ref_points, projected):
        if cur_point[0] <= 0 or cur_point[1] <= 0:
            continue
        cv2.arrowedLine(
            images[1], pt1=ref_point, pt2=cur_point.astype(int), color=(0, 0, 255), thickness=2, tipLength=0.1
        )

    cv2.imshow("result", images[1])
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
