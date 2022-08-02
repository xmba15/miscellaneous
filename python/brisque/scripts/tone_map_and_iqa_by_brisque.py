#!/usr/bin/env python
import os

import cv2
import numpy as np

from brisque_wrapper import BrisqueWrapper

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    hdr_image_path = os.path.join(_CURRENT_DIR, "../data/Market3.hdr")
    hdr_float32 = cv2.imread(hdr_image_path, cv2.IMREAD_ANYDEPTH)
    hdr_float32 = np.clip(hdr_float32, 0, 1)

    # https://learnopencv.com/high-dynamic-range-hdr-imaging-using-opencv-cpp-python/
    tone_maps = [
        cv2.createTonemapDrago(1.0, 0.7),
        cv2.createTonemapReinhard(1.5, 0, 0, 0),
        cv2.createTonemapMantiuk(2.2, 0.85, 1.2),
    ]
    ldr_8bits = [tone_map.process(hdr_float32) * 255 for tone_map in tone_maps]
    ldr_8bits = [ldr_8bit.astype(np.uint8) for ldr_8bit in ldr_8bits]

    wrapper = BrisqueWrapper()
    brisque_scores = [
        wrapper.brisque.compute(image_8bit)[0] for image_8bit in ldr_8bits
    ]

    print(brisque_scores)


if __name__ == "__main__":
    main()
