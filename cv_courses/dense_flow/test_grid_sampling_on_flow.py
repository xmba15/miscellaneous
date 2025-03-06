import cv2
import numpy as np
import torch
import torch.nn.functional as F


def get_args():
    import argparse

    parser = argparse.ArgumentParser("dense optical flow")
    parser.add_argument("--query_path", type=str, default="./data/car1.jpg")
    parser.add_argument("--ref_path", type=str, default="./data/car2.jpg")

    return parser.parse_args()


def flow_to_grid(flow):
    """Convert optical flow to a sampling grid for grid_sample."""
    h, w = flow.shape[:2]

    # Create normalized coordinate grid (H, W)
    grid_y, grid_x = np.meshgrid(
        np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij"
    )

    # Normalize flow to [-1, 1] range
    flow_x = (flow[..., 0] / w) * 2  # Normalize by width
    flow_y = (flow[..., 1] / h) * 2  # Normalize by height

    # Add flow to grid to get the sampling coordinates
    grid = np.stack((grid_x + flow_x, grid_y + flow_y), axis=-1)

    # Convert to PyTorch tensor and reshape for grid_sample
    grid = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W, 2)

    return grid


def main():
    args = get_args()

    query_image = cv2.imread(args.query_path)
    ref_image = cv2.imread(args.ref_path)
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    flow = cv2.optflow.calcOpticalFlowDenseRLOF(query_image, ref_image, None)

    grid = flow_to_grid(flow)

    query_tensor = (
        torch.tensor(query_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    )
    warped_image = F.grid_sample(
        query_tensor, grid, mode="bilinear", align_corners=True
    )
    warped_image_np = (
        warped_image.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    )
    cv2.imwrite("warped.jpg", warped_image_np)


if __name__ == "__main__":
    main()
