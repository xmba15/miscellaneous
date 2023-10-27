import cv2
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia_moons.viz import draw_LAF_matches
from scipy import optimize


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


def main():
    image_paths = [
        "./data/car1.jpg",
        "./data/car2.jpg",
    ]
    images = [cv2.imread(image_path) for image_path in image_paths]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    disk = KF.DISK.from_pretrained("depth").to(device)
    features_list = [extract_features(image, disk, device) for image in images]
    kps0, descs0 = features_list[0].keypoints, features_list[0].descriptors
    kps1, descs1 = features_list[1].keypoints, features_list[1].descriptors
    lafs0 = KF.laf_from_center_scale_ori(
        kps0[None], 96 * torch.ones(1, len(kps0), 1, 1, device=device)
    )
    lafs1 = KF.laf_from_center_scale_ori(
        kps1[None], 96 * torch.ones(1, len(kps1), 1, 1, device=device)
    )

    lightglue_matcher = KF.LightGlueMatcher("disk").to(device)
    dists, idxs = lightglue_matcher(descs0, descs1, lafs0, lafs1)

    mkpts0, mkpts1 = get_matching_keypoints(kps0, kps1, idxs)

    M, mask = cv2.findHomography(
        mkpts0.detach().cpu().numpy(),
        mkpts1.detach().cpu().numpy(),
        cv2.USAC_MAGSAC,
        5.0,
    )
    warped = cv2.warpPerspective(
        images[0], M, images[1].shape[:2][::-1], flags=cv2.INTER_CUBIC
    )
    warped = cv2.addWeighted(images[1], 0.5, warped, 0.5, 2.2)
    cv2.imwrite("warped.jpg", warped)

    optimized_M = refine_homography(
        M,
        mkpts0.detach().cpu().numpy(),
        mkpts1.detach().cpu().numpy(),
    )
    optimized_warped = cv2.warpPerspective(
        images[0], optimized_M, images[1].shape[:2][::-1], flags=cv2.INTER_CUBIC
    )
    optimized_warped = cv2.addWeighted(images[1], 0.5, optimized_warped, 0.5, 2.2)
    cv2.imwrite("optimized_warped.jpg", optimized_warped)

    print("homography before refinement: \n", M)
    print("homography after refinement: \n", optimized_M)

    # # fundamental matrix
    # Fm, inliers = cv2.findFundamentalMat(
    #     mkpts0.detach().cpu().numpy(),
    #     mkpts1.detach().cpu().numpy(),
    #     cv2.USAC_MAGSAC,
    #     1.0,
    #     0.999,
    #     100000,
    # )
    # inliers = inliers > 0
    # print(f"{inliers.sum()} inliers with lightglue")

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # draw_LAF_matches(
    #     KF.laf_from_center_scale_ori(kps0[None].cpu()),
    #     KF.laf_from_center_scale_ori(kps1[None].cpu()),
    #     idxs.cpu(),
    #     images[0],
    #     images[1],
    #     inliers,
    #     draw_dict={
    #         "inlier_color": (0.2, 1, 0.2),
    #         "tentative_color": (1, 1, 0.2, 0.3),
    #         "feature_color": None,
    #         "vertical": False,
    #     },
    #     ax=ax,
    # )
    # plt.savefig("result.png")


def optimize_func(h, query_kpts, ref_kpts):
    H = np.reshape(h, (3, 3))
    transformed_query_pts = cv2.perspectiveTransform(query_kpts[None, ...], H).squeeze(
        0
    )
    diff0 = transformed_query_pts - ref_kpts

    H_inv = np.linalg.inv(H)
    H_inv /= H_inv[2, 2]
    transformed_ref_pts = cv2.perspectiveTransform(ref_kpts[None, ...], H_inv).squeeze(
        0
    )
    diff1 = transformed_ref_pts - query_kpts

    all_diff = np.concatenate([diff0, diff1], axis=0)
    return all_diff.transpose(1, 0).ravel()


def refine_homography(H, query_kpts, ref_kpts):
    sol = optimize.least_squares(
        fun=optimize_func, x0=H.ravel(), args=(query_kpts, ref_kpts), method="lm"
    )
    optimized_H = sol.x.reshape(3, 3)
    optimized_H /= optimized_H[2, 2]

    return optimized_H


if __name__ == "__main__":
    main()
