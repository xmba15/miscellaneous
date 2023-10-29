from dataclasses import dataclass
from typing import List, Optional

import cv2
import kornia.feature as KF
import numpy as np
import torch
from scipy import optimize

from build_panorama import build_panorama


@dataclass
class MatchGroup:
    query_kpts: List[cv2.KeyPoint]
    ref_kpts: List[cv2.KeyPoint]
    query_idx: int
    ref_idx: int
    homography: Optional[np.ndarray]


def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]]
    mkpts2 = kp2[idxs[:, 1]]
    return mkpts1, mkpts2


def extract_features(image: np.ndarray, detector, device):
    with torch.no_grad():
        return detector(
            (torch.tensor(image.transpose(2, 0, 1)[None, ...]).float() / 255.0).to(
                device
            ),
            pad_if_not_divisible=True,
        )[0]


def get_lafs(kpts, device):
    return KF.laf_from_center_scale_ori(
        kpts[None], 96 * torch.ones(1, len(kpts), 1, 1, device=device)
    )


def get_cv2_kpts(kpts_tensor):
    return [
        cv2.KeyPoint(x=x, y=y, size=0.0)
        for (x, y) in kpts_tensor.detach().cpu().numpy()
    ]


def match_keypoints(matcher, features0, features1, device, query_idx, ref_idx):
    kpts0, descs0 = features0.keypoints, features0.descriptors
    kpts1, descs1 = features1.keypoints, features1.descriptors

    dists, idxs = matcher(
        descs0,
        descs1,
        get_lafs(kpts0, device),
        get_lafs(kpts1, device),
    )

    mkpts0, mkpts1 = get_matching_keypoints(kpts0, kpts1, idxs)
    M, mask = cv2.findHomography(
        mkpts0.detach().cpu().numpy(),
        mkpts1.detach().cpu().numpy(),
        cv2.USAC_MAGSAC,
        5.0,
    )

    kpts0 = get_cv2_kpts(mkpts0)
    kpts1 = get_cv2_kpts(mkpts1)

    return MatchGroup(kpts0, kpts1, query_idx, ref_idx, M)


def main():
    image_paths = [
        "./data/starry_night_1.jpg",
        "./data/starry_night_2.jpg",
        "./data/starry_night_3.jpg",
        "./data/starry_night_4.jpg",
    ]
    images = [cv2.imread(image_path) for image_path in image_paths]
    for image in images:
        if image is None:
            print(image)
    grays = [cv2.imread(image_path, 0) for image_path in image_paths]
    pair_indices = [(1, 0), (2, 1), (3, 2)]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    disk_detector = KF.DISK.from_pretrained("depth").to(device)
    features_list = [extract_features(image, disk_detector, device) for image in images]

    lightglue_matcher = KF.LightGlueMatcher("disk").to(device)

    match_groups = []
    for query_idx, ref_idx in pair_indices:
        match_groups.append(
            match_keypoints(
                lightglue_matcher,
                features_list[query_idx],
                features_list[ref_idx],
                device,
                query_idx,
                ref_idx,
            )
        )

    cur_homography = np.eye(3, 3, dtype=np.float64)
    homographies = []
    for idx, match_group in enumerate(match_groups):
        cur_homography = cur_homography.dot(match_group.homography)
        cur_homography /= cur_homography[2, 2]
        homographies.append(cur_homography.copy())

    print("before refinement:")
    for homography in homographies:
        print(homography)

    panorama = build_panorama(images[0], images[1:], homographies)
    cv2.imwrite("panorama.jpg", panorama)

    print("after refinement:")
    homographies_params = []
    for homography in homographies:
        homographies_params += homography.ravel().tolist()
    homographies_params = np.array(homographies_params)

    sol = optimize.least_squares(
        fun=_optimize_func, x0=homographies_params, args=(match_groups,), method="lm"
    )

    refined_homographies = get_homographies_from_sol(sol.x)
    for homography in refined_homographies:
        print(homography)

    refined_panorama = build_panorama(images[0], images[1:], refined_homographies)
    cv2.imwrite("refined_panorama.jpg", refined_panorama)


def get_homographies_from_sol(solution):
    num_homographies = len(solution) // 9
    homographies = []
    for idx in range(num_homographies):
        homography = np.array(solution[idx * 9 : idx * 9 + 9]).reshape(3, 3)
        homography /= homography[2, 2]
        homographies.append(homography)

    return homographies


def _optimize_func(_homographies_params, match_groups):
    num_homographies = len(_homographies_params) // 9
    homographies = []
    for idx in range(num_homographies):
        homography = np.array(_homographies_params[idx * 9 : idx * 9 + 9]).reshape(3, 3)
        homography /= homography[2, 2]
        homographies.append(homography)

    all_diffs = []
    for match_group in match_groups:
        query_idx = match_group.query_idx
        ref_idx = match_group.ref_idx

        if ref_idx == 0:
            H_query_to_ref = homographies[query_idx - 1]
            H_ref_to_query = np.linalg.inv(H_query_to_ref)
        else:
            H_query_to_ref = np.linalg.inv(homographies[ref_idx - 1]).dot(
                homographies[query_idx - 1]
            )
            H_ref_to_query = np.linalg.inv(homographies[query_idx - 1]).dot(
                homographies[ref_idx - 1]
            )

        H_query_to_ref /= H_query_to_ref[2, 2]
        H_ref_to_query /= H_ref_to_query[2, 2]

        transformed_query_pts = cv2.perspectiveTransform(
            np.float64([kpt.pt for kpt in match_group.query_kpts])[None, ...],
            H_query_to_ref,
        ).squeeze(0)
        diff0 = transformed_query_pts - np.float64(
            [kpt.pt for kpt in match_group.ref_kpts]
        )

        transformed_ref_pts = cv2.perspectiveTransform(
            np.float64([kpt.pt for kpt in match_group.ref_kpts])[None, ...],
            H_ref_to_query,
        ).squeeze(0)
        diff1 = transformed_ref_pts - np.float64(
            [kpt.pt for kpt in match_group.query_kpts]
        )
        all_diffs.append(diff0)
        all_diffs.append(diff1)

    all_diff = np.concatenate(all_diffs, axis=0)

    return all_diff.transpose(1, 0).ravel()


if __name__ == "__main__":
    main()
