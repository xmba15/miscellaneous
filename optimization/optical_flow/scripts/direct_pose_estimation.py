#!/usr/bin/env python
import cv2
import numpy as np
from scipy.optimize import least_squares
from liegroups.numpy import SE3
import numba as nb
from new_types import CameraMatrix
from typing import TypeVar, Generic, List, Any, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(np.ndarray, Generic[Shape, DType]):
    pass


def get_subpixels(
    img: Array[Tuple[int, int], np.uint8],
    points: Array[Tuple[int, Literal[2]], np.float64],
) -> Array[Tuple[int], np.float64]:
    height, width = img.shape[:2]
    x, y = tuple(points.T)
    x[x < 0] = 0
    y[y < 0] = 0
    x[x >= width - 1] = width - 2
    y[y >= height - 1] = height - 2
    x_x = x - np.floor(x)
    y_y = y - np.floor(y)
    x_floor = x.astype(int)
    y_floor = y.astype(int)
    x_ceil = np.minimum(np.ones_like(x) * (width - 1), np.ceil(x)).astype(int)
    y_ceil = np.minimum(np.ones_like(y) * (height - 1), np.ceil(y)).astype(int)

    return (
        (1 - x_x) * (1 - y_y) * img[y_floor, x_floor]
        + x_x * (1 - y_y) * img[y_floor, x_ceil]
        + (1 - x_x) * y_y * img[y_ceil, x_floor]
        + x_x * y_y * img[y_ceil, x_ceil]
    )


def get_derivative_xs(
    img: Array[Tuple[int, int], np.uint8],
    points: Array[Tuple[int, Literal[2]], np.float64],
) -> Array[Tuple[int], np.float64]:
    return 0.5 * (
        get_subpixels(img, np.array((points[:, 0] + 1, points[:, 1])).T)
        - get_subpixels(img, np.array((points[:, 0] - 1, points[:, 1])).T)
    )


def get_derivative_ys(
    img: Array[Tuple[int, int], np.uint8],
    points: Array[Tuple[int, Literal[2]], np.float64],
) -> Array[Tuple[int], np.float64]:
    return 0.5 * (
        get_subpixels(img, np.array((points[:, 0], points[:, 1] + 1)).T)
        - get_subpixels(img, np.array((points[:, 0], points[:, 1] - 1)).T)
    )


def get_image_pyramid(
    img: Array[Tuple[int, int], np.uint8], num_scale: int = 4, scale_factor: float = 0.5
) -> List[Array[Tuple[int, int], np.uint8]]:
    pyramid = list()
    for i in range(num_scale):
        if i == 0:
            pyramid.append(img)
        else:
            cur_scale_img = cv2.GaussianBlur(pyramid[i - 1], (5, 5), 0)
            pyramid.append(
                cv2.resize(cur_scale_img, dsize=None, fx=scale_factor, fy=scale_factor)
            )

    return pyramid


class DirectPoseEstimationHandler:
    def __init__(
        self, img1, img2, ref_points, ref_points_3d, camera_matrix, half_window_size
    ):
        self._img1 = img1
        self._img2 = img2
        self._ref_points = ref_points
        self._ref_points_3d = ref_points_3d
        self._camera_matrix = camera_matrix
        self._half_window_size = half_window_size
        self._patch_size = (2 * self._half_window_size + 1) * (
            2 * self._half_window_size + 1
        )
        self._height, self._width = self._img1.shape[:2]

        self._ref_points_patch_values = (
            DirectPoseEstimationHandler.get_image_patch_values(
                self._img1, self._ref_points, self._half_window_size
            )
        )

        self._cur_points_3d = np.empty_like(self._ref_points_3d, dtype=np.float64)
        self._invalids = np.empty(len(self._ref_points), dtype=np.bool)

        # projections of reference points in image 2
        self._projecteds = np.empty_like(self._ref_points, dtype=np.float64)

    @staticmethod
    def get_all_patch_points(points, half_window_size):
        patch_size = (2 * half_window_size + 1) ** 2
        x_s, y_s = np.meshgrid(
            range(-half_window_size, half_window_size + 1),
            range(-half_window_size, half_window_size + 1),
        )
        xy_s = np.array([x_s.flatten(), y_s.flatten()]).T
        all_points = np.tile(xy_s, (len(points), 1)) + np.repeat(
            points, repeats=patch_size, axis=0
        )
        return all_points

    @staticmethod
    def get_image_patch_values(img, points, half_window_size):
        return get_subpixels(
            img,
            DirectPoseEstimationHandler.get_all_patch_points(points, half_window_size),
        )

    @staticmethod
    @nb.njit(cache=True)
    def project_points(camera_matrix, points_3d):
        projecteds = np.empty((len(points_3d), 2), dtype=np.float64)
        projecteds[:, 0] = (
            camera_matrix.fx * points_3d[:, 0] / points_3d[:, 2] + camera_matrix.cx
        )
        projecteds[:, 1] = (
            camera_matrix.fy * points_3d[:, 1] / points_3d[:, 2] + camera_matrix.cy
        )

        return projecteds

    def objective_func(self, coeffs):
        self._cur_points_3d = SE3.exp(coeffs).dot(self._ref_points_3d)
        self._invalids = self._cur_points_3d[:, 2] < 0

        self._projecteds = DirectPoseEstimationHandler.project_points(
            self._camera_matrix, self._cur_points_3d
        )

        # flag array to show which point  does not contribute to jacobian.
        # jacobian at that point is all 0.
        self._invalids |= (
            (self._projecteds[:, 0] < self._half_window_size)
            | (self._projecteds[:, 0] > self._width - self._half_window_size)
            | (self._projecteds[:, 1] < self._half_window_size)
            | (self._projecteds[:, 1] > self._height - self._half_window_size)
        )

        projected_patch_values = DirectPoseEstimationHandler.get_image_patch_values(
            self._img2, self._projecteds, self._half_window_size
        )
        residuals = self._ref_points_patch_values - projected_patch_values
        residuals[np.repeat(self._invalids, repeats=self._patch_size, axis=0)] = 0

        return residuals

    @staticmethod
    @nb.njit(cache=True)
    def get_jacobian_pixel_xi(camera_matrix, point_3d):
        p3d_x = point_3d[0]
        p3d_y = point_3d[1]
        p3d_z = point_3d[2]
        p3d_z_inv = 1.0 / p3d_z
        p3d_z2_inv = p3d_z_inv * p3d_z_inv

        j_pixel_xi = np.empty((2, 6), dtype=np.float64)
        j_pixel_xi[0, 0] = camera_matrix.fx * p3d_z_inv
        j_pixel_xi[0, 1] = 0
        j_pixel_xi[0, 2] = -camera_matrix.fx * p3d_x * p3d_z2_inv
        j_pixel_xi[0, 3] = -camera_matrix.fx * p3d_x * p3d_y * p3d_z2_inv
        j_pixel_xi[0, 4] = (
            camera_matrix.fx + camera_matrix.fx * p3d_x * p3d_x * p3d_z2_inv
        )
        j_pixel_xi[0, 5] = -camera_matrix.fx * p3d_y * p3d_z_inv
        j_pixel_xi[1, 0] = 0
        j_pixel_xi[1, 1] = camera_matrix.fy * p3d_z_inv
        j_pixel_xi[1, 2] = -camera_matrix.fy * p3d_y * p3d_z2_inv
        j_pixel_xi[1, 3] = (
            -camera_matrix.fy - camera_matrix.fy * p3d_y * p3d_y * p3d_z2_inv
        )
        j_pixel_xi[1, 4] = camera_matrix.fy * p3d_x * p3d_y * p3d_z2_inv
        j_pixel_xi[1, 5] = camera_matrix.fy * p3d_x * p3d_z_inv

        return j_pixel_xi

    @staticmethod
    def get_jacobian_pixel_xis(camera_matrix, points_3d):
        return np.array(
            [
                DirectPoseEstimationHandler.get_jacobian_pixel_xi(
                    camera_matrix, point_3d
                )
                for point_3d in points_3d
            ]
        )

    def jacobian(self, coeffs):
        projected_patch_points = DirectPoseEstimationHandler.get_all_patch_points(
            self._projecteds, self._half_window_size
        )
        all_j_img_pixel = np.array(
            (
                get_derivative_xs(self._img2, projected_patch_points),
                get_derivative_ys(self._img2, projected_patch_points),
            )
        ).T
        all_j_img_pixel = all_j_img_pixel.reshape(
            len(self._ref_points), self._patch_size, 2
        )

        projected_jacobian_pixel_xis = (
            DirectPoseEstimationHandler.get_jacobian_pixel_xis(
                self._camera_matrix, self._cur_points_3d
            )
        )

        # depth-wise multiplication of num_points:patch:2 with num_points:2:6
        jacobian = -1.0 * np.matmul(
            all_j_img_pixel, projected_jacobian_pixel_xis
        ).reshape(-1, 6)

        jacobian[
            np.repeat(self._invalids, repeats=self._patch_size, axis=0)
        ] = np.zeros(6, jacobian.dtype)

        return jacobian

    @property
    def projecteds(self):
        assert self._projecteds is not None, "null projected points"
        return self._projecteds


def calc_direct_pose_estimation_single_layer(
    img1, img2, ref_points, ref_points_depth, K, pose21_SE3, half_window_size: int = 1
):
    ref_points_3d = cv2.hconcat(
        (
            (ref_points[:, 0] - K.cx) / K.fx,
            (ref_points[:, 1] - K.cy) / K.fy,
            np.ones(len(ref_points)),
        )
    )
    ref_points_3d *= ref_points_depth[:, None]

    handler = DirectPoseEstimationHandler(
        img1, img2, ref_points, ref_points_3d, K, half_window_size
    )

    pose21_se3 = pose21_SE3.log()
    res_lsq = least_squares(
        handler.objective_func,
        pose21_se3,
        method="lm",
        verbose=2,
        max_nfev=20,
        jac=handler.jacobian,
    )

    return SE3.exp(res_lsq.x), handler.projecteds


def calc_direct_pose_estimation_multi_layer(
    img1,
    img2,
    ref_points,
    ref_points_depth,
    camera_matrix,
    pose21_SE3,
    half_window_size: int = 1,
    num_scale: int = 4,
    scale_factor: float = 0.4,
):
    scales = [1.0]
    for i in range(1, num_scale):
        scales.append(scales[-1] * scale_factor)

    pyr1 = get_image_pyramid(img1, num_scale, scale_factor)
    pyr2 = get_image_pyramid(img2, num_scale, scale_factor)

    projecteds: Array[Tuple[int, Literal[2]], np.float64]
    for i in reversed(range(num_scale)):
        pose21_SE3, projecteds = calc_direct_pose_estimation_single_layer(
            pyr1[i],
            pyr2[i],
            ref_points * scales[i],
            ref_points_depth,
            camera_matrix.scale(scales[i]),
            pose21_SE3,
            half_window_size,
        )
        print("T21={}".format(pose21_SE3))

    return pose21_SE3, projecteds
