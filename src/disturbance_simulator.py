"""Disturbance, measurement noise, and occlusion simulator for Part 2.

This module provides:
- Camera measurement noise injection (Gaussian noise, sensor jitter, outlier spikes).
- Temporary beacon dropouts (occlusions, optical interference, field-of-view exit).
- External physical disturbances (sudden accelerations, wind gusts, drift).
- Generation of synthetic BeaconMeasurement frames matching the Part 1 interface.
"""

from typing import Optional, Tuple
import numpy as np

from config import DisturbanceConfig
from src.tracking_state import BeaconMeasurement


class DisturbanceSimulator:
    """Injects optical dropouts, measurement noise, and dynamic disturbances."""

    def __init__(self, config: Optional[DisturbanceConfig] = None) -> None:
        """Initialize disturbance and occlusion parameters.
        
        Args:
            config: Disturbance configuration settings.
        """
        self.config = config or DisturbanceConfig()
        self.current_frame: int = 0

    def apply_disturbances(
        self,
        ground_truth_pos: Tuple[float, float],
        timestamp: float,
        frame_idx: int
    ) -> BeaconMeasurement:
        """Process ground-truth position through noise and occlusion models to produce synthetic optical measurement.
        
        Args:
            ground_truth_pos: True (x, y) beacon position.
            timestamp: Simulation timestamp in seconds.
            frame_idx: Current simulation frame counter.
            
        Returns:
            BeaconMeasurement representing simulated camera detection from Part 1.
        """
        # Skeleton: check occlusion schedule and inject Gaussian measurement noise
        raise NotImplementedError("DisturbanceSimulator.apply_disturbances skeleton - to be implemented in Phase 2.")

    def is_occluded(self, frame_idx: int) -> bool:
        """Determine if the beacon is occluded in the given frame based on configured intervals.
        
        Args:
            frame_idx: Current frame index.
            
        Returns:
            True if beacon is currently occluded / undetected.
        """
        if not self.config.enable_occlusions:
            return False
        for start_f, end_f in self.config.occlusion_intervals:
            if start_f <= frame_idx <= end_f:
                return True
        return False

    def add_measurement_noise(self, true_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Add zero-mean Gaussian noise to simulate camera sensor jitter.
        
        Args:
            true_pos: True (x, y) coordinates.
            
        Returns:
            Noisy (x, y) coordinates.
        """
        if not self.config.enable_measurement_noise:
            return true_pos
        noise = np.random.normal(0.0, self.config.measurement_noise_std, size=2)
        return (true_pos[0] + float(noise[0]), true_pos[1] + float(noise[1]))

    def reset(self) -> None:
        """Reset the disturbance simulator frame state."""
        self.current_frame = 0
