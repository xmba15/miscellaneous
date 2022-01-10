#!/usr/bin/env python
import argparse

import cv2

from yolox_detector import YoloxDetector


def get_args():
    parser = argparse.ArgumentParser("test onnx detector")
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.45)
    parser.add_argument("--image_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()
    labels = args.labels.split(",")
    yolox_detector = YoloxDetector(
        onnx_model_path=args.onnx_model_path,
        input_shape=(640, 640),
        with_p6=False,
        labels=labels,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
    )
    image = cv2.imread(args.image_path)
    assert image is not None, f"failed to read {args.image_path}"
    detections = yolox_detector.detect(image)
    image = yolox_detector.draw_detections(image, detections)
    cv2.imshow("detection result", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
