#!/usr/bin/env python
from typing import NamedTuple


__all__ = ["CameraMatrix"]


class CameraMatrix(NamedTuple):
    fx: float
    fy: float
    cx: float
    cy: float
    base_line: float

    def scale(self, scale_factor: float) -> "CameraMatrix":
        assert scale_factor > 0, "scale factor must be > 0"
        return CameraMatrix(
            self.fx * scale_factor,
            self.fy * scale_factor,
            self.cx * scale_factor,
            self.cy * scale_factor,
            self.base_line,
        )
