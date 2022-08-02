import os
import subprocess
import tempfile
from typing import Optional

import cv2


class BrisqueWrapper:
    __MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_contrib/9d0a451bee4cdaf9d3f76912e5abac6000865f1a/modules/quality/samples/brisque_model_live.yml"
    __RANGE_URL = "https://raw.githubusercontent.com/opencv/opencv_contrib/9d0a451bee4cdaf9d3f76912e5abac6000865f1a/modules/quality/samples/brisque_range_live.yml"

    def __init__(
        self, model_path: Optional[str] = None, range_path: Optional[str] = None
    ):
        self._model_path = model_path
        self._range_path = range_path
        self._temp_dir = tempfile.TemporaryDirectory()

        if self._model_path is None:
            cmd = f"wget {self.__MODEL_URL} -P {self._temp_dir.name}"
            subprocess.run(cmd, shell=True)
            self._model_path = os.path.join(
                self._temp_dir.name,
                os.path.basename(self.__MODEL_URL),
            )

        if self._range_path is None:
            cmd = f"wget {self.__RANGE_URL} -P {self._temp_dir.name}"
            subprocess.run(cmd, shell=True)
            self._range_path = os.path.join(
                self._temp_dir.name,
                os.path.basename(self.__RANGE_URL),
            )

        self._brisque = cv2.quality.QualityBRISQUE_create(
            self._model_path, self._range_path
        )

    @property
    def brisque(self):
        return self._brisque

    def __del__(self):
        self._temp_dir.cleanup()
