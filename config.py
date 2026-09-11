"""Configuration settings for Part 2: Predictive Tracking & Disturbance Handling.

This module defines all tunable parameters for:
- Kalman Filter (process & measurement noise, covariance matrices)
- Tracking State Machine (confidence thresholds, loss timeouts)
- Beacon Motion & Disturbance Simulators (noise levels, occlusion schedules)
- Visualizer & Demonstration settings (window size, frame rates, color schemes)
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class KalmanConfig:
    """Kalman filter tuning parameters."""
    dt: float = 1.0 / 30.0  # Nominal time step (30 FPS)
    process_noise_std_pos: float = 0.5  # Standard deviation of position process noise
    process_noise_std_vel: float = 1.0  # Standard deviation of velocity process noise
    measurement_noise_std: float = 2.5  # Standard deviation of optical measurement noise (pixels)
    initial_covariance_pos: float = 10.0  # Initial position uncertainty
    initial_covariance_vel: float = 100.0  # Initial velocity uncertainty


@dataclass
class TrackingStateConfig:
    """State machine confidence thresholds and timeouts."""
    min_confidence_to_lock: float = 0.7  # Confidence required to enter TRACKING_LOCKED
    max_frames_lost_before_search: int = 15  # Frames without detection before entering SEARCH_ACQUISITION
    max_frames_lost_before_lost: int = 60  # Frames without detection before marking LOST
    reacquisition_frames_required: int = 3  # Consecutive detection frames required for reacquisition
    confidence_decay_rate: float = 0.05  # Per-frame confidence decay during missing detections


@dataclass
class DisturbanceConfig:
    """Disturbance and noise simulation parameters."""
    enable_measurement_noise: bool = True
    measurement_noise_std: float = 3.0  # Measurement noise standard deviation
    enable_occlusions: bool = True
    occlusion_intervals: list = field(default_factory=lambda: [(60, 100), (180, 240)])  # Frame ranges for occlusion
    enable_wind_drift: bool = True
    wind_drift_magnitude: float = 0.8  # Disturbance drift acceleration


@dataclass
class BeaconSimConfig:
    """Ground truth beacon simulation parameters."""
    arena_width: int = 1280
    arena_height: int = 720
    trajectory_type: str = "circular"  # "linear", "circular", "figure8", "random_walk"
    base_speed: float = 5.0
    start_pos: Tuple[float, float] = (640.0, 360.0)


@dataclass
class VisualizerConfig:
    """Visualization window and overlay styling."""
    window_name: str = "SIH Part 2 - Predictive Tracking & Disturbance Handling"
    canvas_width: int = 1280
    canvas_height: int = 720
    fps: int = 30
    trail_length: int = 64
    show_covariance_ellipse: bool = True
    show_ground_truth: bool = True


@dataclass
class SystemConfig:
    """Global system configuration aggregating all component configurations."""
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    tracking: TrackingStateConfig = field(default_factory=TrackingStateConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    beacon: BeaconSimConfig = field(default_factory=BeaconSimConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)


# Default system configuration instance
DEFAULT_CONFIG = SystemConfig()
