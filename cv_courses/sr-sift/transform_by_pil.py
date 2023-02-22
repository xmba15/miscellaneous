import argparse

import cv2
import numpy as np
import skimage
from codetiming import Timer
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser("Test PIL Image perspective transform")
    parser.add_argument("--query_image_path", "-q", type=str, required=True)
    parser.add_argument("--ref_image_path", "-r", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()
    query_image = cv2.imread(args.query_image_path)
    ref_image = cv2.imread(args.ref_image_path)
    assert query_image is not None
    assert ref_image is not None

    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    detector = cv2.SIFT_create(nfeatures=8000)
    query_kpts, query_descs = detector.detectAndCompute(query_gray, None)
    ref_kpts, ref_descs = detector.detectAndCompute(ref_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(query_descs, ref_descs)
    M, _ = cv2.findHomography(
        np.float64([query_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
        np.float64([ref_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
        cv2.RANSAC,
        5.0,
    )

    with Timer(text="warping by opencv: {:.4f} seconds"):
        warped_opencv = cv2.warpPerspective(
            query_image, M, ref_image.shape[:2][::-1], flags=cv2.INTER_CUBIC
        )

    with Timer(text="warping by scikit-image: {:.4f} seconds"):
        warped_skimage = skimage.transform.warp(
            query_image,
            np.linalg.inv(M),
            output_shape=ref_image.shape[:2],
            preserve_range=True,
            order=3,
        ).astype(np.uint8)

    with Timer(text="warping by PIL: {:.4f} seconds"):
        inv_homography = np.linalg.inv(M)
        inv_homography /= inv_homography[2, 2]
        coeffs = inv_homography.flatten()
        warped_pil = np.asarray(
            Image.fromarray(query_image).transform(
                ref_image.shape[:2][::-1], Image.PERSPECTIVE, coeffs, Image.BICUBIC
            )
        )

    cv2.imwrite(
        "warped_opencv.jpg", cv2.addWeighted(ref_image, 0.5, warped_opencv, 0.5, 2.2)
    )
    cv2.imwrite(
        "warped_skimage.jpg", cv2.addWeighted(ref_image, 0.5, warped_skimage, 0.5, 2.2)
    )
    cv2.imwrite("warped_pil.jpg", cv2.addWeighted(ref_image, 0.5, warped_pil, 0.5, 2.2))


if __name__ == "__main__":
    # scikit-image: 0.19.3
    # Pillow: 9.4.0
    # codetiming: 1.4.0
    # python3 transform_by_pil.py -q ./data/44-l.JPG -r ./data/44-r.JPG
    main()
    # warping by opencv: 0.0033 seconds
    # warping by scikit-image: 0.3884 seconds
    # warping by PIL: 0.0382 seconds"
