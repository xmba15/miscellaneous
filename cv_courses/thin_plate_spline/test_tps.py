#!/usr/bin/env python
import os

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.feature import *

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def load_torch_image(image):
    image = K.image_to_tensor(image, False).float() / 255.0
    return image


def match_detector_free(
    query_gray: np.ndarray,
    ref_gray: np.ndarray,
    conf_thresh=0.99,
):
    matcher = KF.LoFTR(pretrained="outdoor")
    input_dict = {
        "image0": load_torch_image(query_gray),
        "image1": load_torch_image(ref_gray),
    }
    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    conf_mask = np.where(correspondences["confidence"].cpu().numpy() > conf_thresh)
    mkpts0 = mkpts0[conf_mask]
    mkpts1 = mkpts1[conf_mask]

    def _np_to_cv2_kpts(np_kpts):
        cv2_kpts = []
        for np_kpt in np_kpts:
            cur_cv2_kpt = cv2.KeyPoint()
            cur_cv2_kpt.pt = tuple(np_kpt)
            cv2_kpts.append(cur_cv2_kpt)
        return cv2_kpts

    return _np_to_cv2_kpts(mkpts0), _np_to_cv2_kpts(mkpts1)


def main():
    image_paths = (
        os.path.join(_CURRENT_DIR, "data/car1.jpg"),
        os.path.join(_CURRENT_DIR, "data/car2.jpg"),
    )
    images = [cv2.imread(image_path) for image_path in image_paths]
    grays = [cv2.imread(image_path, 0) for image_path in image_paths]

    query_kpts, ref_kpts = match_detector_free(
        grays[0],
        grays[1],
    )
    matches = [
        cv2.DMatch(_queryIdx=idx, _trainIdx=idx, _distance=0)
        for idx in range(len(query_kpts))
    ]

    matched_image = cv2.drawMatches(
        images[0],
        query_kpts,
        images[1],
        ref_kpts,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imwrite("matched_image.jpg", matched_image)

    tps = cv2.createThinPlateSplineShapeTransformer()
    src = np.float32([list(kpt.pt) for kpt in query_kpts]).reshape(1, -1, 2)
    dst = np.float32([list(kpt.pt) for kpt in ref_kpts]).reshape(1, -1, 2)

    print("number of matches: ", len(matches))
    tps.estimateTransformation(dst, src, matches)
    warped = tps.warpImage(grays[0])

    print(warped.max())
    cv2.imwrite("tps_warped.jpg", warped)


if __name__ == "__main__":
    main()
