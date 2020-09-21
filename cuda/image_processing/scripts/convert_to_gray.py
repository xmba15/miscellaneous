#!/usr/bin/env python
import time
import numpy as np
import cv2

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


def _get_args():
    import argparse

    parser = argparse.ArgumentParser("")
    parser.add_argument("--img_path", type=str, required=True)
    args = parser.parse_args()

    return args


def main():
    args = _get_args()
    img = cv2.imread(args.img_path, 1)
    assert img is not None, "empty image"
    height, width = img.shape[:2]
    img = img.astype(np.float32)

    convert_to_gray_kernel = """
    __global__ void convertToGray(const float* rgb, unsigned char* gray, int width, int height)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x < width && y < height) {
            int grayOffset = y * width + x;
            int rgbOffset = grayOffset * 3;
            gray[grayOffset] = (unsigned char)(0.299f * rgb[rgbOffset + 2] + 0.587f * rgb[rgbOffset + 1] + 0.114f * rgb[rgbOffset]);
        }
    }
    """

    device_img = cuda.mem_alloc(img.nbytes)
    cuda.memcpy_htod(device_img, img)

    # or
    # device_img = gpuarray.to_gpu(img.astype(np.float32))

    gray = np.empty(img.shape[:2], np.uint8)
    device_gray = cuda.mem_alloc(gray.nbytes)

    # or
    # device_gray = gpuarray.GPUArray(img.shape[:2], np.uint8)

    mod = SourceModule(convert_to_gray_kernel)
    convert_to_gray_func = mod.get_function("convertToGray")

    block_dim = (32, 32, 1)
    grid_dim = ((width + block_dim[0] - 1) // block_dim[0], (height + block_dim[1] - 1) // block_dim[1])

    time_sta = time.perf_counter()
    convert_to_gray_func(device_img, device_gray, np.int32(width), np.int32(height), block=block_dim, grid=grid_dim)

    # or use numpy array directly
    # convert_to_gray_func(cuda.In(img), cuda.Out(gray), np.int32(width), np.int32(height), block=block_dim, grid=grid_dim)

    time_end = time.perf_counter()
    processing_time = time_end - time_sta

    cuda.memcpy_dtoh(gray, device_gray)

    # or (if GPUArray is used)
    # gray = device_gray.get()

    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("processing time: {}[ms]".format(processing_time * 1000))


if __name__ == "__main__":
    main()
