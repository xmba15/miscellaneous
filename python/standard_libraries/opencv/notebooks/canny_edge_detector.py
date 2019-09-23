#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import enum
from scipy.ndimage.filters import convolve as convolveimg


class THRESHOLD_PIXEL(enum.Enum):
    NON_RELEVANT = 0
    STRONG = 1
    WEAK = 2


class CannyEdgeDetector(object):
    def __init__(self, gray_img):
        assert gray_img.ndim == 2
        assert isinstance(gray_img, np.ndarray)
        assert gray_img.dtype == np.float32
        self.gray_img = gray_img

    def blur_img(self, img, kernel_size, sigma):
        def __gaussian_kernel(kernel_size, sigma):
            assert isinstance(kernel_size, int)
            assert kernel_size % 2 == 1
            half_size = kernel_size / 2
            x, y = np.mgrid[-half_size : half_size + 1, -half_size : half_size + 1]
            normal = 1 / (2.0 * np.pi * sigma ** 2)
            g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
            return g

        return cv2.filter2D(img, -1, __gaussian_kernel(kernel_size, sigma))

    def sobel_filter(self, img):
        def __sobel_x(img):
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)

            I_x = convolveimg(img, kernel_x)
            return I_x

        def __sobel_y(img):
            kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

            I_y = convolveimg(img, kernel_y)
            return I_y

        I_x = __sobel_x(img)
        I_y = __sobel_y(img)

        I = np.hypot(I_x, I_y)

        assert I.max() > 0
        I = I / I.max() * 255
        theta = np.arctan2(I_y, I_x)
        return (I, theta)

    def non_max_suppression(self, gradient_img, theta):
        nms = np.copy(gradient_img)
        HEIGHT, WIDTH = gradient_img.shape

        PI_over_8 = np.pi / 8

        for i in range(1, HEIGHT - 1):
            for j in range(1, WIDTH - 1):
                direction = theta[i, j]
                if (0 <= direction < PI_over_8) or (
                    15 * PI_over_8 <= direction <= 2 * np.pi
                ):
                    q = gradient_img[i, j + 1]
                    r = gradient_img[i, j - 1]
                elif (PI_over_8 <= direction < 3 * PI_over_8) or (
                    9 * PI_over_8 <= direction < 11 * PI_over_8
                ):
                    q = gradient_img[i + 1, j - 1]
                    r = gradient_img[i - 1, j + 1]
                elif (3 * PI_over_8 <= direction < 5 * PI_over_8) or (
                    11 * PI_over_8 <= direction < 13 * PI_over_8
                ):
                    q = gradient_img[i + 1, j]
                    r = gradient_img[i - 1, j]
                else:
                    q = gradient_img[i - 1, j - 1]
                    r = gradient_img[i + 1, j + 1]

                if (gradient_img[i, j] < q) or (gradient_img[i, j] < r):
                    nms[i, j] = 0

        return nms

    def thresholding(self, img, low_threshold_ratio, high_threshold_ratio):
        high_threshold = img.max() * high_threshold_ratio
        low_threshold = high_threshold * low_threshold_ratio

        res = np.zeros(img.shape, dtype=np.uint8)

        res[np.where(img >= high_threshold)] = THRESHOLD_PIXEL.STRONG.value
        res[
            np.where((img <= high_threshold) & (img >= low_threshold))
        ] = THRESHOLD_PIXEL.WEAK.value

        return res

    def hysteresis(self, thresholded):
        def eight_neighbors(row, col, rows, cols):
            indices = []
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    if i == 0 and j == 0:
                        continue
                    row_idx = row + i
                    col_idx = col + j
                    if (row_idx < 0 or row_idx > rows - 1) or (
                        col_idx < 0 or col_idx > cols - 1
                    ):
                        continue
                    indices.append((row_idx, col_idx))
            return indices

        HEIGHT, WIDTH = thresholded.shape

        hysteresis_thresholded = np.copy(thresholded)
        for ir in range(HEIGHT):
            for ic in range(WIDTH):
                if thresholded[ir, ic] != THRESHOLD_PIXEL.WEAK.value:
                    continue

                for idx in eight_neighbors(ir, ic, HEIGHT, WIDTH):
                    if thresholded[idx[0], idx[1]] == THRESHOLD_PIXEL.STRONG.value:
                        hysteresis_thresholded[ir, ic] = THRESHOLD_PIXEL.STRONG.value
                        break

                if hysteresis_thresholded[ir, ic] == THRESHOLD_PIXEL.WEAK.value:
                    hysteresis_thresholded[ir, ic] = THRESHOLD_PIXEL.NON_RELEVANT.value

        assert (
            np.where(hysteresis_thresholded == THRESHOLD_PIXEL.STRONG.value)[0].size > 0
        )
        assert (
            np.where(hysteresis_thresholded == THRESHOLD_PIXEL.NON_RELEVANT.value)[
                0
            ].size
            > 0
        )
        return hysteresis_thresholded

    def hysteresis_img(self, thresholded, strong_value=255):
        hysteresis_thresholded = self.hysteresis(thresholded)
        result = np.copy(hysteresis_thresholded)
        result[
            np.where(hysteresis_thresholded == THRESHOLD_PIXEL.STRONG.value)
        ] = strong_value

        assert np.where(result == THRESHOLD_PIXEL.WEAK.value)[0].size == 0
        return result

    def edge_detection(
        self,
        strong_value=255,
        low_threshold_ratio=0.07,
        high_threshold_ratio=0.19,
        kernel_size=5,
        sigma=1,
    ):
        blurred_img = self.blur_img(self.gray_img, kernel_size, sigma)
        gradient_img, theta = self.sobel_filter(blurred_img)
        nms_img = self.non_max_suppression(gradient_img, theta)
        thresholded = self.thresholding(
            nms_img, low_threshold_ratio, high_threshold_ratio
        )

        edge_img = self.hysteresis_img(thresholded, strong_value)

        return edge_img
