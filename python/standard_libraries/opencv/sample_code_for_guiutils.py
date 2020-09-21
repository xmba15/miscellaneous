#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import cv2


class EdgeFinder:
    def __init__(self, image, filter_size=1, threshold1=0, threshold2=0):
        self.image = image
        self._filter_size = filter_size
        self._threshold1 = threshold1
        self._threshold2 = threshold2

        def onchangeThreshold1(pos):
            self._threshold1 = pos
            self.__render()

        def onchangeThreshold2(pos):
            self._threshold2 = pos
            self.__render()

        def onchangeFilterSize(pos):
            self._filter_size = pos
            self._filter_size += (self._filter_size + 1) % 2
            self.__render()

        cv2.namedWindow('edges')

        cv2.createTrackbar('threshold1', 'edges',
                           self._threshold1, 255, onchangeThreshold1)
        cv2.createTrackbar('threshold2', 'edges',
                           self._threshold2, 255, onchangeThreshold2)
        cv2.createTrackbar('filter_size', 'edges',
                           self._filter_size, 20, onchangeFilterSize)

        self.__render()

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    def threshold1(self):
        return self._threshold1

    def threshold2(self):
        return self._threshold2

    def filterSize(self):
        return self._filter_size

    def edgeImage(self):
        return self._edge_img

    def smoothedImage(self):
        return self._smoothed_img

    def __render(self):
        self._smoothed_img = cv2.GaussianBlur(
            self.image, (self._filter_size, self._filter_size), sigmaX=0, sigmaY=0)
        self._edge_img = cv2.Canny(
            self._smoothed_img, self._threshold1, self._threshold2)
        cv2.imshow('smoothed', self._smoothed_img)
        cv2.imshow('edges', self._edge_img)


def main():

    if (len(sys.argv) != 2):
        print("Usage: python sample_code_for_guiutils.py [path to image]")
        exit(0)

    img = cv2.imread(sys.argv[1], 0)

    if (img is None):
        print("image empty")
        print("Usage: python sample_code_for_guiutils.py [path to image]")
        exit(0)

    edge_finder = EdgeFinder(img)


if __name__ == '__main__':
    main()
