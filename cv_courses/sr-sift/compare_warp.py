import cv2
import kornia as K
import kornia.geometry as KG
import numpy as np
import skimage
import torch.nn.functional as F


def get_sift_based_matching_homography(query_image, ref_image):
    detector = cv2.SIFT_create()
    query_kpts, query_descs = detector.detectAndCompute(query_image, None)
    ref_kpts, ref_descs = detector.detectAndCompute(ref_image, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(query_descs, ref_descs)

    M, _ = cv2.findHomography(
        np.float64([query_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
        np.float64([ref_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
        cv2.RANSAC,
        5.0,
    )
    warped = cv2.warpPerspective(
        query_image, M, ref_image.shape[:2][::-1], flags=cv2.INTER_CUBIC
    )
    warped = cv2.addWeighted(ref_image, 0.5, warped, 0.5, 2.2)

    return warped


def get_sift_based_matching_ecc_refine_homography(query_image, ref_image):
    detector = cv2.SIFT_create()
    query_kpts, query_descs = detector.detectAndCompute(query_image, None)
    ref_kpts, ref_descs = detector.detectAndCompute(ref_image, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(query_descs, ref_descs)

    M, _ = cv2.findHomography(
        np.float64([query_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
        np.float64([ref_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
        cv2.RANSAC,
        5.0,
    )
    number_of_iterations = 5000
    termination_eps = 1e-5
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps,
    )

    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    warped = cv2.warpPerspective(
        query_image, M, ref_image.shape[:2][::-1], flags=cv2.INTER_CUBIC
    )
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_mask = warped_gray > 0
    _, M = cv2.findTransformECC(
        ref_gray,
        warped_gray,
        np.eye(3, 3, dtype=np.float32),
        cv2.MOTION_HOMOGRAPHY,
        criteria,
        warped_mask.astype(np.uint8) * 255,
    )

    print("ecc matrix: ", M)

    warped = cv2.warpPerspective(
        warped, M, ref_image.shape[:2][::-1], flags=cv2.INTER_CUBIC
    )
    warped = cv2.addWeighted(ref_image, 0.5, warped, 0.5, 2.2)

    return warped


def get_sift_based_matching_kornia_direct_matching_refine_homography(
    query_image, ref_image
):
    detector = cv2.SIFT_create()
    query_kpts, query_descs = detector.detectAndCompute(query_image, None)
    ref_kpts, ref_descs = detector.detectAndCompute(ref_image, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(query_descs, ref_descs)

    M, _ = cv2.findHomography(
        np.float64([query_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
        np.float64([ref_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
        cv2.RANSAC,
        5.0,
    )

    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    warped = cv2.warpPerspective(
        query_image, M, ref_image.shape[:2][::-1], flags=cv2.INTER_CUBIC
    )
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    registrator = KG.ImageRegistrator(
        "homography",
        loss_fn=F.mse_loss,
        lr=8e-4,
        pyramid_levels=3,
        num_iterations=5000,
        tolerance=1e-5,
    )

    def load_timage(gray):
        return K.image_to_tensor(gray, None).float() / 255.0

    M = registrator.register(load_timage(warped_gray), load_timage(ref_gray))

    print("torch direct optimization: ", M)

    warped = KG.homography_warp(
        load_timage(warped), M, load_timage(ref_gray).shape[-2:]
    )
    warped = K.tensor_to_image(warped * 255).astype(np.uint8)
    warped = cv2.addWeighted(ref_image, 0.3, warped, 0.7, 2.2)

    return warped


def get_sift_based_matching_skimage_warp_homography(query_image, ref_image):

    detector = cv2.SIFT_create()
    query_kpts, query_descs = detector.detectAndCompute(query_image, None)
    ref_kpts, ref_descs = detector.detectAndCompute(ref_image, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(query_descs, ref_descs)

    M, _ = cv2.findHomography(
        np.float64([query_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
        np.float64([ref_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
        cv2.RANSAC,
        5.0,
    )
    warped = skimage.transform.warp(
        query_image,
        np.linalg.inv(M),
        output_shape=ref_image.shape[:2],
        preserve_range=True,
    ).astype(np.uint8)

    # warped = cv2.warpPerspective(
    #     query_image, M, ref_image.shape[:2][::-1], flags=cv2.INTER_CUBIC
    # )

    warped = cv2.addWeighted(ref_image, 0.5, warped, 0.5, 2.2)

    return warped


def main():
    image1_path = "data/44-l.JPG"
    image2_path = "data/44-r.JPG"
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    warped1 = get_sift_based_matching_homography(image1, image2)
    warped2 = get_sift_based_matching_ecc_refine_homography(image1, image2)
    warped3 = get_sift_based_matching_skimage_warp_homography(image1, image2)
    warped4 = get_sift_based_matching_kornia_direct_matching_refine_homography(
        image1, image2
    )

    cv2.imshow("warped_sift", warped1)
    cv2.imshow("warped_sift_ecc", warped2)
    cv2.imshow("warped_sift_skimage_warp", warped3)
    cv2.imshow("warped_sift_kornia_direct_opt", warped4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
