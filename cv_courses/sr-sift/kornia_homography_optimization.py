from typing import List

import cv2
import kornia as K
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def load_image(image: np.ndarray) -> torch.Tensor:
    tensor: torch.Tensor = K.utils.image_to_tensor(image).float() / 255.0  # CxHxW
    return tensor[None]  # 1xCxHxW


def get_gaussian_pyramid(img: torch.Tensor, num_levels: int) -> List[torch.Tensor]:
    pyramid = []
    pyramid.append(img)
    for _ in range(num_levels - 1):
        img_curr = pyramid[-1]
        img_down = K.geometry.pyrdown(img_curr)
        pyramid.append(img_down)
    return pyramid


class Homography(nn.Module):
    def __init__(self) -> None:
        super(Homography, self).__init__()
        self.homography = nn.Parameter(torch.Tensor(3, 3))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.eye_(self.homography)

    def forward(self) -> torch.Tensor:
        return torch.unsqueeze(self.homography, dim=0)  # 1x3x3


def compute_scale_loss(
    img_src: torch.Tensor,
    img_dst: torch.Tensor,
    dst_homo_src: nn.Module,
    optimizer: torch.optim,
    num_iterations: int,
    error_tol: float,
) -> torch.Tensor:
    assert len(img_src.shape) == len(img_dst.shape), (img_src.shape, img_dst.shape)

    # init loop parameters
    loss_tol = torch.tensor(error_tol)
    loss_prev = torch.finfo(img_src.dtype).max

    for i in range(num_iterations):
        # create homography warper
        src_homo_dst: torch.Tensor = torch.inverse(dst_homo_src)

        _height, _width = img_src.shape[-2:]
        warper = K.geometry.HomographyWarper(_height, _width)
        img_src_to_dst = warper(img_src, src_homo_dst)

        # compute and mask loss
        loss = F.l1_loss(img_src_to_dst, img_dst, reduction="none")  # 1x3xHxW

        ones = warper(torch.ones_like(img_src), src_homo_dst)
        loss = loss.masked_select((ones > 0.9)).mean()

        # compute gradient and update optimizer parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    learning_rate: float = 1e-4  # the gradient optimisation update step
    num_iterations: int = 1000  # the number of iterations until convergence
    num_levels: int = 6  # the total number of image pyramid levels
    error_tol: float = 1e-8

    image1_path = "data/44-l.JPG"
    image2_path = "data/44-r.JPG"
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    detector = cv2.SIFT_create()
    query_kpts, query_descs = detector.detectAndCompute(image1, None)
    ref_kpts, ref_descs = detector.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(query_descs, ref_descs)

    M, _ = cv2.findHomography(
        np.float64([query_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
        np.float64([ref_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
        cv2.RANSAC,
        5.0,
    )
    warped = cv2.warpPerspective(
        image1, M, image2.shape[:2][::-1], flags=cv2.INTER_CUBIC
    )
    cv2.imshow("warped", cv2.addWeighted(image2, 0.5, warped, 0.5, 2.2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # image1_tmp = np.zeros_like(image2)
    # image1_tmp[: image1.shape[0], : image1.shape[1]] = image1

    image1_tensor = load_image(warped)
    image2_tensor = load_image(image2)

    img_src_pyr: List[torch.Tensor] = get_gaussian_pyramid(image1_tensor, num_levels)
    img_dst_pyr: List[torch.Tensor] = get_gaussian_pyramid(image2_tensor, num_levels)

    dst_homo_src = Homography()

    optimizer = optim.Adam(dst_homo_src.parameters(), lr=learning_rate)

    for iter_idx in range(num_levels):
        scale: int = (num_levels - 1) - iter_idx
        img_src = img_src_pyr[scale]
        img_dst = img_dst_pyr[scale]

        # compute scale loss
        compute_scale_loss(
            img_src, img_dst, dst_homo_src(), optimizer, num_iterations, error_tol
        )

        print("Optimization iteration: {}/{}".format(iter_idx, num_levels))

    homography = dst_homo_src.state_dict()["homography"].detach().numpy()
    warped = cv2.warpPerspective(
        image1, homography, image2.shape[:2][::-1], flags=cv2.INTER_CUBIC
    )
    warped = cv2.addWeighted(image2, 0.5, warped, 0.5, 2.2)
    cv2.imshow("warped", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    h, w = img_src.shape[-2:]
    warper = K.geometry.HomographyWarper(h, w)
    img_src_to_dst = warper(img_src, torch.inverse(dst_homo_src()))
    img_src_to_dst_merge = 0.65 * img_src_to_dst + 0.35 * img_dst
    img_src_to_dst_vis = K.utils.tensor_to_image(img_src_to_dst_merge)
    cv2.imshow("warped", img_src_to_dst_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
