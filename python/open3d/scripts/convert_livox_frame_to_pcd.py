#!/usr/bin/env python
import argparse
import numpy as np
import open3d as o3d


def _process_one_frame(frame_path: str) -> np.array:
    num_lines = sum(1 for line in open(frame_path))
    xyz = np.zeros((num_lines, 3), dtype=np.float32)

    count = 0
    with open(frame_path) as f:
        for line in f:
            elements = line.split(",")
            for i in range(3):
                xyz[count, i] = float(elements[i])
            count += 1

    return xyz


def _get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--livox_frame", type=str, required=True)
    args = parser.parse_args()

    return args


def main():
    args = _get_args()

    output_file_name = args.livox_frame[:-4] + ".pcd"

    xyz = _process_one_frame(args.livox_frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(output_file_name, pcd, write_ascii=True)


if __name__ == "__main__":
    main()
