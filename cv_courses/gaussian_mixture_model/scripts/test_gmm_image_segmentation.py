#!/usr/bin/env python
import argparse

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--num_k", type=int, default=5)

    return parser.parse_args()


def main():
    args = get_args()
    image = cv2.imread(args.image_path)
    assert image is not None, f"failed to load {args.image_path}"

    pixel_mean = np.mean(image, axis=(0, 1))
    pixel_std = np.std(image, axis=(0, 1))
    norm_image = (image - pixel_mean) / pixel_std
    gm = GaussianMixture(n_components=args.num_k, random_state=0).fit(
        norm_image.reshape(-1, 3)
    )

    result = gm.predict(norm_image.reshape(-1, 3)).reshape(*image.shape[:2], -1)

    COLORS = [
        (255, 0, 0),  # red
        (0, 255, 0),  # green
        (0, 0, 255),  # blue
        (255, 255, 0),  # yellow
        (255, 0, 255),  # magenta
    ]
    result = np.array(COLORS, dtype=np.uint8)[result]
    result = np.squeeze(result, axis=2)
    print(result.shape)
    cv2.imshow("result", result)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
