"""Configuration settings for Part 2: Predictive Tracking & Disturbance Handling.

This module defines all tunable parameters for:
- Beacon Motion Simulation (initial coordinates, velocity, boundary margins, random seed)
- Disturbance & Occlusion Simulation (noise std, jitter std, occlusion intervals, random seed)
- Tracking State Machine (confidence thresholds, loss timeouts)
- Kalman Filter (process & measurement noise, covariance matrices - for Phase 3)
- Visualizer & Demonstration settings (window size, frame rates, color schemes)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BeaconSimConfig:
    """Ground-truth beacon motion simulation parameters."""
    arena_width: int = 1280
    arena_height: int = 720
    start_pos: Tuple[float, float] = (300.0, 250.0)
    start_velocity: Tuple[float, float] = (4.5, 3.2)  # pixels per frame
    speed_magnitude: float = 5.0                       # nominal speed in pixels per frame
    heading_noise_std: float = 0.04                    # slight steering / trajectory perturbation per frame (rad)
    margin: float = 50.0                               # boundary safety margin to trigger smooth bounce / turn
    random_seed: Optional[int] = 42                    # seed for reproducible ground truth motion


@dataclass
class DisturbanceConfig:
    """Disturbance, measurement noise, and occlusion simulation parameters."""
    enable_measurement_noise: bool = True
    measurement_noise_std: float = 4.0                 # Standard deviation of optical measurement noise (pixels)
    enable_jitter: bool = True
    jitter_std: float = 1.0                            # High-frequency camera jitter noise (pixels)
    enable_occlusions: bool = True
    occlusion_intervals: List[Tuple[int, int]] = field(
        default_factory=lambda: [(60, 110), (180, 230), (300, 350)]
    )                                                  # Frame ranges [start_frame, end_frame] where detection is lost
    default_confidence: float = 0.95                   # Confidence assigned to valid detections
    random_seed: Optional[int] = 101                   # seed for reproducible disturbance / noise generation


@dataclass
class KalmanConfig:
    """Kalman filter tuning parameters (for Phase 3)."""
    dt: float = 1.0 / 30.0                             # Nominal time step (30 FPS)
    process_noise_std_pos: float = 0.5                 # Standard deviation of position process noise
    process_noise_std_vel: float = 1.0                 # Standard deviation of velocity process noise
    measurement_noise_std: float = 4.0                 # Standard deviation of optical measurement noise (pixels)
    initial_covariance_pos: float = 10.0               # Initial position uncertainty
    initial_covariance_vel: float = 100.0              # Initial velocity uncertainty


@dataclass
class TrackingStateConfig:
    """State machine confidence thresholds and timeouts (for Phase 3)."""
    min_confidence_to_lock: float = 0.7
    max_frames_lost_before_search: int = 15
    max_frames_lost_before_lost: int = 60
    reacquisition_frames_required: int = 3
    confidence_decay_rate: float = 0.05


@dataclass
class VisualizerConfig:
    """Visualization window and overlay styling."""
    window_name: str = "SIH Part 2 - Kalman Predictive Tracking (Phase 3)"
    canvas_width: int = 1280
    canvas_height: int = 720
    fps: int = 30
    trail_length: int = 80                             # Number of historical points to display
    show_ground_truth: bool = True
    show_measurements: bool = True


@dataclass
class SystemConfig:
    """Global system configuration aggregating all component configurations."""
    beacon: BeaconSimConfig = field(default_factory=BeaconSimConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    tracking: TrackingStateConfig = field(default_factory=TrackingStateConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)


# Default system configuration instance
DEFAULT_CONFIG = SystemConfig()
