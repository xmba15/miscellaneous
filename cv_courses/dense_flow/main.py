"""
Ref: https://pytorch.org/vision/main/auto_examples/plot_optical_flow.html
"""
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from skimage.transform import warp as sk_warp
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
from torchvision.utils import flow_to_image


def get_args():
    import argparse

    parser = argparse.ArgumentParser("dense optical flow")
    parser.add_argument("--query_path", type=str, required=True)
    parser.add_argument("--ref_path", type=str, required=True)

    return parser.parse_args()


def estimate_optical_flow(query_image, ref_image):
    orig_shape = query_image.shape[:2]

    query_image = torch.from_numpy(query_image.transpose(2, 0, 1))[None, :, :, :]
    ref_image = torch.from_numpy(ref_image.transpose(2, 0, 1))[None, :, :, :]

    # query_image = F.resize(query_image, size=[520, 960])
    # ref_image = F.resize(ref_image, size=[520, 960])

    # weights = Raft_Small_Weights.DEFAULT
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    query_image, ref_image = transforms(query_image, ref_image)

    from torchvision.models.optical_flow import raft_large, raft_small

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = raft_large(weights=weights, progress=False).to(device)
    # model = raft_small(weights=weights, progress=False).to(device)
    model = model.eval()
    list_of_flows = model(ref_image.to(device), query_image.to(device))
    predicted_flows = list_of_flows[-1]
    # predicted_flows = F.resize(predicted_flows, size=orig_shape)

    return predicted_flows.squeeze(0).detach().cpu().numpy()


def main():
    args = get_args()
    query_image = cv2.imread(args.query_path)
    ref_image = cv2.imread(args.ref_path)
    assert query_image is not None and ref_image is not None

    orig_shape = query_image.shape[:2]

    predicted_flows = estimate_optical_flow(query_image, ref_image)
    flow_imgs = flow_to_image(torch.from_numpy(predicted_flows))
    output = flow_imgs.detach().cpu().numpy().transpose(1, 2, 0)

    cv2.imwrite("flow.png", output)

    flow_x, flow_y = predicted_flows
    height, width = orig_shape
    row_coords, col_coords = np.meshgrid(
        np.arange(height), np.arange(width), indexing="ij"
    )
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    query_warp = sk_warp(
        query_gray,
        np.array([row_coords + flow_y, col_coords + flow_x]),
        mode="edge",
        preserve_range=True,
    ).astype(np.uint8)

    cv2.imwrite("query_warp.png", query_warp)


if __name__ == "__main__":
    main()
