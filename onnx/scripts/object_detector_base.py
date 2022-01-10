#!/usr/bin/env python
from typing import Generic, List, Optional, Tuple, TypeVar

import cv2
import numpy as np

try:
    from typing import Annotated, Literal  # type: ignore
except ImportError:
    from typing_extensions import Annotated, Literal


Shape = TypeVar("Shape")
DType = TypeVar("DType")


__all__ = ["Array", "Detection", "BaseDetector"]


class Array(np.ndarray, Generic[Shape, DType]):
    pass


class Detection:
    def __init__(self, xyxy: Annotated[Tuple[float], 4], label_id: int, conf: float):
        self._xyxy = xyxy
        assert len(self._xyxy) == 4, "wrong size of tuple for detection position"
        assert self._xyxy[0] < self._xyxy[2] and self._xyxy[1] < self._xyxy[3], "invalid detection"

        self._label_id = label_id
        self._conf = conf

    @property
    def xywh(self):
        xmin, ymin, xmax, ymax = self._xyxy
        return (xmin, ymin, xmax - xmin, ymax - ymin)

    @property
    def xyxy(self):
        return self._xyxy

    @property
    def label_id(self):
        return self._label_id

    @property
    def conf(self):
        return self._conf


class BaseDetector:
    def __init__(
        self,
        labels: Tuple[str],
        conf_thresh: float = 0.5,
        colors: Optional[List[Annotated[List[int], 3]]] = None,
        seed: int = 2022,
    ):
        assert len(labels) > 0, "empty labels"
        self._labels = labels
        np.random.seed(seed)
        self._colors = (
            colors if colors is not None else np.random.choice(range(256), size=(len(self._labels), 3)).tolist()
        )
        assert len(self._labels) == len(self._colors), "mismatch sizes of labels and colors"
        self._conf_thresh = conf_thresh

    def draw_detections(
        self, image: Array[Tuple[int, int, Literal[3]], np.uint8], detections: List[Detection], thickness: int = 2
    ) -> Array[Tuple[int, int, Literal[3]], np.uint8]:
        visualized = np.copy(image)
        for detection in detections:
            pt1 = tuple(map(int, detection.xyxy[:2]))
            pt2 = tuple(map(int, detection.xyxy[2:]))
            cv2.rectangle(
                visualized,
                pt1=pt1,
                pt2=pt2,
                color=self._colors[detection.label_id],
                thickness=thickness,
            )
            label = f"{self._labels[detection.label_id]} {round(detection.conf,2)}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(
                visualized,
                pt1=pt1,
                pt2=(pt1[0] + label_size[0], pt1[1] + int(1.3 * label_size[1])),
                color=self._colors[detection.label_id],
            )
            cv2.putText(
                visualized, label, (pt1[0], pt1[1] + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255)
            )

        return visualized
