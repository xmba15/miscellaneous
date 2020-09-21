#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pycocotools.mask as mask_util
import numpy as np
from PIL import Image
import cv2


COCO_CLASSES = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize(
        (int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR
    )

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype("float32")

    print(image.shape)

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math

    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, : image.shape[1], : image.shape[2]] = image

    print(padded_image.shape)
    return padded_image


def display_objdetect_image(
    image, boxes, labels, scores, masks, classes, score_threshold=0.7
):
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio

    _, ax = plt.subplots(1, figsize=(12, 9))

    image = np.array(image)

    for mask, box, label, score in zip(masks, boxes, labels, scores):
        # Showing boxes with score > 0.7
        if score <= score_threshold:
            continue

        # Finding contour based on mask
        mask = mask[0, :, :, None]
        int_box = [int(i) for i in box]
        mask = cv2.resize(
            mask, (int_box[2] - int_box[0] + 1, int_box[3] - int_box[1] + 1)
        )
        mask = mask > 0.5
        im_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        x_0 = max(int_box[0], 0)
        x_1 = min(int_box[2] + 1, image.shape[1])
        y_0 = max(int_box[1], 0)
        y_1 = min(int_box[3] + 1, image.shape[0])
        mask_y_0 = max(y_0 - box[1], 0)
        mask_y_1 = mask_y_0 + y_1 - y_0
        mask_x_0 = max(x_0 - box[0], 0)
        mask_x_1 = mask_x_0 + x_1 - x_0
        im_mask[y_0:y_1, x_0:x_1] = mask[mask_y_0:mask_y_1, mask_x_0:mask_x_1]
        im_mask = im_mask[:, :, None]

        contours, hierarchy = cv2.findContours(
            im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        image = cv2.drawContours(image, contours, -1, 25, 3)

        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )
        ax.annotate(
            classes[label] + ":" + str(np.round(score, 2)),
            (box[0], box[1]),
            color="w",
            fontsize=12,
        )
        ax.add_patch(rect)

    ax.imshow(image)
    plt.show()


def main(args):
    img = Image.open(args.img_path)
    img = img.resize((800, 800))
    img_data = preprocess(img)

    import onnxruntime as rt

    sess = rt.InferenceSession(args.onnx_weight)

    assert len(sess.get_inputs()) == 1
    assert len(sess.get_outputs()) == 4

    input_name = sess.get_inputs()[0].name
    output_names = [elem.name for elem in sess.get_outputs()]
    output = sess.run(output_names, {input_name: img_data})
    boxes, labels, scores, masks = output

    # print(boxes)
    # print(labels)
    # print(scores)
    print(len(boxes))

    display_objdetect_image(
        img, boxes, labels, scores, masks, COCO_CLASSES, args.confidence
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("simple script to test maskrcnn")
    parser.add_argument("--img_path", type=str, required=True, help="path to image")
    parser.add_argument(
        "--onnx_weight", type=str, required=True, help="path to onnx weight"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="class confidence"
    )
    args = parser.parse_args()

    main(args)
