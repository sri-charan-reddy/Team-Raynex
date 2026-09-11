"""Tracking state definitions and data structures for Part 2.

This module defines:
- TrackingState: Finite State Machine (FSM) states for beacon tracking
- BeaconMeasurement: Standard input container representing optical detection data from Part 1
- TrackingResult: Standard output container provided to downstream controllers (Part 4)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np


class TrackingState(Enum):
    """Operational states of the predictive tracking finite state machine."""
    UNINITIALIZED = auto()       # No detections received yet; awaiting initial lock
    TRACKING_LOCKED = auto()     # Receiving regular valid detections; high confidence
    PREDICTING_COASTING = auto() # Missing optical detection; estimating position using Kalman motion model
    SEARCH_ACQUISITION = auto()  # Prolonged loss; expanding search uncertainty area for reacquisition
    LOST = auto()                # Search timeout exceeded; tracking lost, requires re-initialization


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
        confidence: Normalized tracking confidence score in [0.0, 1.0].
        is_predicted: True if this result is purely predicted without a fresh measurement.
        covariance: 2x2 or 4x4 covariance matrix representing state uncertainty.
        frames_without_detection: Number of consecutive frames without measurement.
    """
    timestamp: float
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    state: TrackingState
    confidence: float
    is_predicted: bool
    covariance: Optional[np.ndarray] = None
    frames_without_detection: int = 0
