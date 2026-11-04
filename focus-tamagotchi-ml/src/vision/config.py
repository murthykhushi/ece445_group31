from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math


@dataclass(frozen=True)
class Calibration:
    close_left: float = 0.19
    open_left: float = 0.23
    close_right: float = 0.19
    open_right: float = 0.23
    camera_matrix: list | None = None
    distortion: list | None = None
    image_size: list | None = None  # [width, height], required with intrinsics
    neutral_angles: tuple = (0.0, 0.0, 0.0)  # pitch, yaw, roll
    calibrated_eyes: bool = False
    calibration_id: str = "default-uncalibrated"

    def __post_init__(self):
        for close, opened in ((self.close_left, self.open_left), (self.close_right, self.open_right)):
            if not (math.isfinite(close) and math.isfinite(opened) and 0 < close < opened < 1):
                raise ValueError("EAR thresholds must satisfy 0 < close < open < 1")
        if len(self.neutral_angles) != 3 or not all(math.isfinite(x) for x in self.neutral_angles):
            raise ValueError("neutral_angles must contain finite pitch, yaw, roll")
        if self.camera_matrix is not None:
            import numpy as np
            k = np.asarray(self.camera_matrix, dtype=float)
            if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                    or not np.allclose(k[2], [0, 0, 1])):
                raise ValueError("Invalid calibrated 3x3 camera_matrix")
            if self.image_size is None or len(self.image_size) != 2 or min(self.image_size) <= 0:
                raise ValueError("Calibrated image_size [width, height] is required")
        if self.distortion is not None:
            if len(self.distortion) not in (4, 5, 8, 12, 14) or not all(math.isfinite(x) for x in self.distortion):
                raise ValueError("Invalid distortion coefficients")

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        data['calibration_id'] = hashlib.sha256(raw).hexdigest()[:16]
        return cls(**data)
