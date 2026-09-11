"""Disturbance, measurement noise, and occlusion simulator for Part 2.

This module provides:
- Camera measurement noise injection (zero-mean Gaussian noise on x and y).
- High-frequency camera jitter disturbance simulation.
- Configurable temporary detection loss / occlusion schedules.
- Clean optical measurement generation matching the Part 1 -> Part 2 interface contract.
"""

from typing import Optional, Tuple
import numpy as np

from config import DisturbanceConfig
from src.tracking_state import BeaconMeasurement


class DisturbanceSimulator:
    """Simulates optical measurement noise, sensor jitter, and temporary detection loss."""

    def __init__(self, config: Optional[DisturbanceConfig] = None) -> None:
        """Initialize disturbance and occlusion parameters with a dedicated RNG.
        
        Args:
            config: Disturbance configuration settings.
        """
        self.config = config or DisturbanceConfig()
        self.rng = np.random.RandomState(self.config.random_seed)
        self.current_frame: int = 0

    def is_occluded(self, frame_idx: int) -> bool:
        """Check if the beacon detection is currently lost/occluded for the given frame index.
        
        Args:
            frame_idx: Current simulation frame number.
            
        Returns:
            True if detection is lost during this frame, False otherwise.
        """
        if not self.config.enable_occlusions:
            return False
        for start_f, end_f in self.config.occlusion_intervals:
            if start_f <= frame_idx <= end_f:
                return True
        return False

    def add_measurement_noise(self, true_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Add Gaussian sensor noise and camera jitter to the ground-truth position.
        
        Args:
            true_pos: True (x, y) beacon coordinates.
            
        Returns:
            Noisy (x, y) coordinates representing optical detection.
        """
        meas_x, meas_y = true_pos[0], true_pos[1]

        # 1. Optical sensor measurement noise
        if self.config.enable_measurement_noise and self.config.measurement_noise_std > 0:
            noise = self.rng.normal(0.0, self.config.measurement_noise_std, size=2)
            meas_x += float(noise[0])
            meas_y += float(noise[1])

        # 2. Camera mechanical jitter noise
        if self.config.enable_jitter and self.config.jitter_std > 0:
            jitter = self.rng.normal(0.0, self.config.jitter_std, size=2)
            meas_x += float(jitter[0])
            meas_y += float(jitter[1])

        return (meas_x, meas_y)

    def apply_disturbances(
        self,
        ground_truth_pos: Tuple[float, float],
        timestamp: float,
        frame_idx: int
    ) -> BeaconMeasurement:
        """Produce an optical BeaconMeasurement from ground-truth position.
        
        If the frame falls within an occlusion interval:
            - measured_pos is None
            - detected is False
            - confidence is 0.0
        Otherwise:
            - measured_pos contains noisy (x, y)
            - detected is True
            - confidence is default_confidence
        
        Args:
            ground_truth_pos: True (x, y) beacon position.
            timestamp: Timestamp in seconds.
            frame_idx: Current simulation frame number.
            
        Returns:
            BeaconMeasurement data structure.
        """
        self.current_frame = frame_idx
        occluded = self.is_occluded(frame_idx)

        if occluded:
            return BeaconMeasurement(
                timestamp=timestamp,
                position=None,
                detected=False,
                confidence=0.0,
                raw_bbox=None
            )

        noisy_pos = self.add_measurement_noise(ground_truth_pos)
        return BeaconMeasurement(
            timestamp=timestamp,
            position=noisy_pos,
            detected=True,
            confidence=self.config.default_confidence,
            raw_bbox=None
        )

    def get_measurement(
        self,
        ground_truth_pos: Tuple[float, float],
        frame_idx: int
    ) -> Tuple[Optional[Tuple[float, float]], bool]:
        """Convenience method returning raw (measurement_pos, detection_available) pair.
        
        Args:
            ground_truth_pos: True (x, y) beacon position.
            frame_idx: Current frame index.
            
        Returns:
            Tuple of (measured_position or None, detection_available flag).
        """
        if self.is_occluded(frame_idx):
            return None, False
        noisy_pos = self.add_measurement_noise(ground_truth_pos)
        return noisy_pos, True

    def reset(self) -> None:
        """Reset the disturbance simulator and its internal RNG."""
        self.rng = np.random.RandomState(self.config.random_seed)
        self.current_frame = 0
