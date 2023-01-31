import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from skimage.transform import warp as sk_warp


def get_args():
    import argparse

    parser = argparse.ArgumentParser("test keynet")
    parser.add_argument("--image_path1", type=str, required=True)
    parser.add_argument("--image_path2", type=str, required=True)

    return parser.parse_args()


def detector_free_match(query_gray, ref_gray):
    matcher = KF.LoFTR(pretrained="outdoor")
    input_dict = {
        "image0": load_torch_image(query_gray),
        "image1": load_torch_image(ref_gray),
    }
    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    conf_mask = np.where(correspondences["confidence"].cpu().numpy() > 0.5)
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
    args = get_args()
    query_image = cv2.imread(args.image_path1)
    ref_image = cv2.imread(args.image_path2)
    assert query_image is not None
    assert ref_image is not None

    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    query_kpts, ref_kpts = detector_free_match(query_gray, ref_gray)

    matched_image = cv2.drawMatches(
        query_image,
        query_kpts,
        ref_image,
        ref_kpts,
        [
            cv2.DMatch(_queryIdx=idx, _trainIdx=idx, _distance=0)
            for idx in range(len(query_kpts))
        ],
        None,
        flags=2,
    )
    cv2.imwrite("loftr_matched_image.jpg", matched_image)

    query_features = compute_features(query_gray, query_kpts)
    ref_features = compute_features(ref_gray, ref_kpts)

    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(
        query_features[1], ref_features[1]
    )
    matches = sorted(matches, key=lambda x: x.distance)
    query_indices, ref_indices = get_match_indices(matches)
    query_features = get_keypoints_by_indices(*query_features, query_indices)
    ref_features = get_keypoints_by_indices(*ref_features, ref_indices)

    dists, idxs = KF.match_adalam(
        torch.tensor(query_features[1]).squeeze(0),
        torch.tensor(ref_features[1]).squeeze(0),
        torch.tensor(query_features[2]),
        torch.tensor(ref_features[2]),
        hw1=torch.tensor(torch.tensor(query_image[:2])),
        hw2=torch.tensor(torch.tensor(ref_image[:2])),
    )

    matches = []
    for idx in idxs.detach().cpu().numpy():
        match = cv2.DMatch()
        match.queryIdx = idx[0]
        match.trainIdx = idx[1]
        matches.append(match)
    print(f"matches: {len(matches)}")
    matches = sorted(matches, key=lambda x: x.distance)
    query_indices, ref_indices = get_match_indices(matches)
    query_features = get_keypoints_by_indices(*query_features, query_indices)
    ref_features = get_keypoints_by_indices(*ref_features, ref_indices)

    query_kpts = query_features[0]
    ref_kpts = ref_features[0]
    H, mask = cv2.findHomography(
        np.float64([query_kpt.pt for query_kpt in query_kpts]).reshape(-1, 1, 2),
        np.float64([ref_kpt.pt for ref_kpt in ref_kpts]).reshape(-1, 1, 2),
        ransacReprojThreshold=0.5,
        maxIters=10000,
        confidence=0.99,
    )
    query_kpts = list(np.array(query_kpts)[np.all(mask > 0, axis=1)])
    ref_kpts = list(np.array(ref_kpts)[np.all(mask > 0, axis=1)])
    print(f"matches after homography estimation: {len(matches)}")

    matched_image = cv2.drawMatches(
        query_image,
        query_kpts,
        ref_image,
        ref_kpts,
        [
            cv2.DMatch(_queryIdx=idx, _trainIdx=idx, _distance=0)
            for idx in range(len(query_kpts))
        ],
        None,
        flags=2,
    )

    cv2.imwrite("matched_image.jpg", matched_image)

    overlaid_img = sk_warp(
        query_image,
        np.linalg.pinv(H),
        output_shape=ref_image.shape[:2],
        preserve_range=True,
    ).astype(np.uint8)

    overlaid_img = cv2.addWeighted(ref_image, 0.5, overlaid_img, 0.5, 2.2)
    cv2.imwrite("overlaid_image.jpg", overlaid_img)


def get_keypoints_by_indices(kpts, descs, lafs, indices):
    return list(np.array(kpts)[indices]), descs[indices, :], lafs[:, indices, :, :]


def get_match_indices(matches):
    query_indices = []
    ref_indices = []
    for cur_match in matches:
        query_indices.append(cur_match.queryIdx)
        ref_indices.append(cur_match.trainIdx)
    return query_indices, ref_indices


def get_default_detector_config():
    import math

    return {
        # Extraction Parameters
        "nms_size": 15,
        "pyramid_levels": 4,
        "up_levels": 1,
        "scale_factor_levels": math.sqrt(2),
        "s_mult": 22.0,
    }


def compute_features(image_gray, kpts):
    image_tensor = load_torch_image(image_gray)
    num_kpts = len(kpts)
    xy = torch.tensor([(x.pt[0], x.pt[1]) for x in kpts], dtype=torch.float).view(
        1, num_kpts, 2
    )
    all_kpts = []
    all_descs = None
    all_lafs = None

    config = get_default_detector_config()
    affnet = KF.LAFAffNetShapeEstimator(True).eval()
    hardnet = KF.HardNet8(True)

    all_factors = []
    for idx_level in range(config["up_levels"]):
        up_factor = config["scale_factor_levels"] ** (1 + idx_level)
        all_factors.append(up_factor)
    cur_factor = 1
    for idx_level in range(config["pyramid_levels"] + 1):
        all_factors.append(cur_factor)
        cur_factor /= config["scale_factor_levels"]

    for up_factor in all_factors:
        all_kpts += kpts
        scales = torch.tensor(
            [config["s_mult"] * up_factor for x in kpts], dtype=torch.float
        ).view(1, num_kpts, 1, 1)

        lafs = K.feature.laf_from_center_scale_ori(xy, scales).reshape(1, -1, 2, 3)
        lafs = affnet(lafs, image_tensor)

        patches = KF.extract_patches_from_pyramid(image_tensor, lafs, 32)
        B, N, CH, H, W = patches.size()
        descs = hardnet(patches.view(B * N, CH, H, W)).view(B * N, -1)
        all_lafs = lafs if all_lafs is None else torch.cat((all_lafs, lafs), 1)
        all_descs = descs if all_descs is None else torch.cat((all_descs, descs), 0)

    return all_kpts, all_descs.detach().cpu().numpy(), all_lafs.detach().cpu().numpy()


def load_torch_image(image):
    image = K.image_to_tensor(image, False).float() / 255.0
    return image


if __name__ == "__main__":
    main()
