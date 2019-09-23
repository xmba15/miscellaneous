#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class HoughLineDetector(object):
    def __init__(self, edge_img):
        assert(edge_img.ndim == 2)
        self.edge_img = edge_img

    def __estimate_hough_accumulator(self):
        HEIGHT, WIDTH = self.edge_img.shape
        img_diagonal_length = int(np.hypot(HEIGHT, WIDTH))

        thetas = np.deg2rad(np.arange(-90, 91))
        ps = np.linspace(-img_diagonal_length,
                         img_diagonal_length, 2 * img_diagonal_length + 1)

        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        num_thetas = len(thetas)
        num_ps = len(ps)

        accumulator = np.zeros((num_ps, num_thetas), dtype=np.uint64)
        ys, xs = np.nonzero(self.edge_img)

        for y, x in zip(ys, xs):
            for i, (cos_val, sin_val) in enumerate(zip(cos_t, sin_t)):
                p_val = int(x * cos_val + y * sin_val) + img_diagonal_length
                accumulator[p_val, i] += 1

        return accumulator, ps, thetas

    def line_from_hough_transform(self, min_voting_num=200, offset=1000):
        accumulator, ps, thetas = self.__estimate_hough_accumulator(
        )

        ps_thetas_indices = np.where(accumulator > min_voting_num)
        lines = []

        for p_idx, theta_idx in zip(ps_thetas_indices[0], ps_thetas_indices[1]):
            p = ps[p_idx]
            theta = thetas[theta_idx]

            # Stores the value of cos(theta) in a
            a = np.cos(theta)

            # Stores the value of sin(theta) in b
            b = np.sin(theta)

            # x0 stores the value pcos(theta)
            x0 = a*p

            # y0 stores the value psin(theta)
            y0 = b*p

            # x1 stores the rounded off value of (pcos(theta)-offset * sin(theta))
            x1 = int(x0 + offset*(-b))

            # y1 stores the rounded off value of (psin(theta)+offset * cos(theta))
            y1 = int(y0 + offset*(a))

            # x2 stores the rounded off value of (pcos(theta)+offset * sin(theta))
            x2 = int(x0 - offset*(-b))

            # y2 stores the rounded off value of (psin(theta)-offset * cos(theta))
            y2 = int(y0 - offset*(a))

            lines.append(((x1, y1), (x2, y2)))

        return lines
