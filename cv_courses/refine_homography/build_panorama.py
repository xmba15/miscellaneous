from typing import List

import cv2
import numpy as np


def get_image_bounds(height, width):
    return np.float64(
        [
            [0, 0],
            [width, 0],
            [0, height],
            [width, height],
        ],
    )


def build_panorama(ref_image, images: List[np.array], homographies: List[np.ndarray]):
    assert len(images) == len(homographies)
    image_bounds_list = [get_image_bounds(*image.shape[:2]) for image in images]
    warped_image_bounds_list = [
        cv2.perspectiveTransform(bound_pts.reshape(-1, 1, 2), homography).reshape(4, 2)
        for (bound_pts, homography) in zip(image_bounds_list, homographies)
    ]
    bounds_max_min = [
        list(pts.max(axis=0).ravel()) + list(pts.min(axis=0).ravel())
        for pts in warped_image_bounds_list
    ]
    bounds_max_min_arr = np.array(bounds_max_min)
    x_min, y_min = bounds_max_min_arr.min(axis=0)[2:]
    x_max, y_max = bounds_max_min_arr.max(axis=0)[:2]

    offset_homography = np.array(
        [
            [1, 0, max(0, -x_min)],
            [0, 1, max(0, -y_min)],
            [0, 0, 1],
        ],
        np.float64,
    )
    new_height = int(np.ceil(y_max - y_min))
    new_width = int(np.ceil(x_max - x_min))
    print(new_height, new_width)

    offset_homographies = [offset_homography] + [
        offset_homography.dot(homography) for homography in homographies
    ]
    warped_images = [
        cv2.warpPerspective(image, homography, (new_width, new_height))
        for (image, homography) in zip([ref_image] + images, offset_homographies)
    ]

    panorama = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    for image in warped_images:
        panorama = np.where(
            np.logical_and(
                np.repeat(np.sum(panorama, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
                np.repeat(np.sum(image, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
            ),
            0,
            image * 0.5 + panorama * 0.5,
        ).astype(np.uint8)

    return panorama
