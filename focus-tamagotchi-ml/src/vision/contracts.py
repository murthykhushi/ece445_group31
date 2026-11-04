from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FramePacket:
    seq: int
    timestamp: float  # monotonic seconds, or source time during deterministic replay
    image: Any
    read_completed: float


@dataclass
class LandmarkPacket:
    points: Any = None  # full-frame (N, 2) pixel coordinates
    schema: str = "ibug68"
    flags: list[str] = field(default_factory=list)
    detection_ran: bool = False
    gpu_ms: float | None = None


@dataclass
class MetricPacket:
    seq: int
    timestamp: float
    backend_actual: str
    ear_left: float | None = None
    ear_right: float | None = None
    eye_state: str = "UNKNOWN"
    closure_fraction_60s: float | None = None
    perclos80_60s: float | None = None  # not populated without anatomical validation
    valid_coverage_60s: float = 0.0
    blink_count_60s: int | None = None
    median_blink_ms: float | None = None
    blink_event_duration_ms: float | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    pose_reprojection_px: float | None = None
    frame_age_ms: float = 0.0
    processing_ms: float = 0.0
    gpu_ms: float | None = None
    detection_ran: bool = False
    calibration_id: str = "default"
    quality_flags: list[str] = field(default_factory=list)
    schema_version: int = 2

    def to_dict(self):
        return asdict(self)
