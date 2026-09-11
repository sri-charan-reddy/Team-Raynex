"""Tracking state definitions and data structures for Part 2.

This module defines:
- TrackingState: Finite State Machine (FSM) states:
    UNINITIALIZED, TRACKING, PREDICTING, SEARCHING, REACQUIRING, LOST
- BeaconMeasurement: Standard input container representing optical detection data from Part 1
- TrackingResult: Standard output container provided to downstream controllers (Part 4)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np


class TrackingState(Enum):
    """Operational states of the predictive tracking finite state machine."""
    UNINITIALIZED = auto()  # No valid initial detection; awaiting initial lock
    TRACKING = auto()       # Valid measurement received; Kalman predict + correct
    PREDICTING = auto()     # Short temporary miss; Kalman predict only (dead reckoning)
    SEARCHING = auto()      # Extended miss; local search scanning around Kalman predicted position
    REACQUIRING = auto()    # Measurement returned after miss/search; passed gating check
    LOST = auto()           # Detection lost longer than max_prediction_frames limit


@dataclass
class BeaconMeasurement:
    """Input data contract representing an optical beacon detection from Part 1.
    
    Attributes:
        timestamp: Monotonic time in seconds when detection occurred.
        position: Measured (x, y) coordinates in pixel space (or camera frame).
        detected: Boolean flag indicating if beacon was detected in the current frame.
        confidence: Optical detector detection confidence score in [0.0, 1.0].
        raw_bbox: Optional bounding box (x, y, w, h) from detector.
    """
    timestamp: float
    position: Optional[Tuple[float, float]] = None
    detected: bool = False
    confidence: float = 0.0
    raw_bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class TrackingResult:
    """Output data contract for downstream controller consumption (Part 4).
    
    Attributes:
        timestamp: Time of the state estimation in seconds.
        position: Estimated / predicted 2D position (x, y).
        velocity: Estimated velocity vector (vx, vy) in pixels/second.
        state: Current TrackingState of the system.
        confidence: Normalized tracking health / confidence score in [0.0, 1.0].
        is_predicted: True if this result is purely dead-reckoned without measurement update.
        miss_count: Number of consecutive frames without an accepted measurement.
        measurement_accepted: True if the current frame measurement passed gating and corrected the filter.
        covariance: 4x4 or 2x2 error covariance matrix representing state uncertainty.
        frames_without_detection: Deprecated alias for miss_count.
    """
    timestamp: float
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    state: TrackingState
    confidence: float
    is_predicted: bool
    miss_count: int = 0
    measurement_accepted: bool = False
    covariance: Optional[np.ndarray] = None
    frames_without_detection: int = 0
