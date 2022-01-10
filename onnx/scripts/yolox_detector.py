#!/usr/bin/env python
from typing import List, Tuple

import cv2
import numpy as np

import onnxruntime as ort
from object_detector_base import Array, BaseDetector, Detection
from utils import multiclass_nms

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal


__all__ = ["YoloxDetector"]


class YoloxDetector(BaseDetector):
    def __init__(
        self,
        onnx_model_path,
        input_shape: Tuple[int, int],
        with_p6: bool,
        labels: Tuple[str],
        conf_thresh: float = 0.5,
        nms_thresh=0.45,
        colors=None,
    ):
        BaseDetector.__init__(self, labels, conf_thresh, colors)
        self._session = ort.InferenceSession(onnx_model_path)
        self._input_shape = input_shape
        self._nms_thresh = nms_thresh
        self._with_p6 = with_p6

    def detect(self, image: Array[Tuple[int, int, Literal[3]], np.uint8]) -> List[Detection]:
        preprocessed, ratio = YoloxDetector.preprocess(image, self._input_shape)
        ort_inputs = {self._session.get_inputs()[0].name: preprocessed[None, :, :, :]}
        output = self._session.run(None, ort_inputs)
        predictions = YoloxDetector.postprocess(output[0], self._input_shape, p6=self._with_p6)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self._nms_thresh, score_thr=self._conf_thresh)
        if dets is None:
            return []
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        return [
            Detection(bbox, int(cls_ind), score)
            for bbox, cls_ind, score in zip(final_boxes, final_cls_inds, final_scores)
        ]

    @staticmethod
    def preprocess(image, input_shape, swap=(2, 0, 1)):
        """
        Ref: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L144
        """
        if len(image.shape) == 3:
            padded_img = np.ones((input_shape[0], input_shape[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_shape, dtype=np.uint8) * 114

        r = min(input_shape[0] / image.shape[0], input_shape[1] / image.shape[1])
        resized_img = cv2.resize(
            image,
            (int(image.shape[1] * r), int(image.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(image.shape[0] * r), : int(image.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    @staticmethod
    def postprocess(outputs, input_shape, p6=False):
        """
        Ref: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py#L99
        """
        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [input_shape[0] // stride for stride in strides]
        wsizes = [input_shape[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            _xv, _yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((_xv, _yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs
