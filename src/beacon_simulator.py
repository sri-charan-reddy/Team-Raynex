"""Beacon ground-truth motion simulator for Part 2 standalone evaluation.

This module provides:
- Ground-truth trajectory generation (circular, linear, figure-8, random walk).
- Simulation of moving optical beacon coordinates over time.
- Emulation of camera field-of-view and frame boundaries.
"""

from typing import Optional, Tuple
import numpy as np

from config import BeaconSimConfig


class BeaconSimulator:
    """Simulates ground-truth motion dynamics of an optical beacon in 2D space."""

    def __init__(self, config: Optional[BeaconSimConfig] = None) -> None:
        """Initialize the beacon simulator with trajectory settings and boundary constraints.
        
        Args:
            config: Beacon motion simulation parameters.
        """
        self.config = config or BeaconSimConfig()
        self.position: np.ndarray = np.array(self.config.start_pos, dtype=np.float64)
        self.velocity: np.ndarray = np.array([self.config.base_speed, 0.0], dtype=np.float64)
        self.frame_count: int = 0
        self.time: float = 0.0

    def step(self, dt: float = 1.0 / 30.0) -> Tuple[float, float]:
        """Advance the simulation by one time step dt and return the new ground truth position.
        
        Args:
            dt: Time increment in seconds.
            
        Returns:
            Tuple of (x, y) ground-truth beacon coordinates.
        """
        # Skeleton: trajectory evolution logic based on configured pattern
        raise NotImplementedError("BeaconSimulator.step skeleton - to be implemented in Phase 2.")

    def get_ground_truth_position(self) -> Tuple[float, float]:
        """Retrieve the current true position of the beacon without advancing time.
        
        Returns:
            Tuple of (x, y) ground-truth beacon coordinates.
        """
        return (float(self.position[0]), float(self.position[1]))

    def get_ground_truth_velocity(self) -> Tuple[float, float]:
        """Retrieve current true velocity vector of the beacon.
        
        Returns:
            Tuple of (vx, vy) ground-truth velocities in pixels/second.
        """
        return (float(self.velocity[0]), float(self.velocity[1]))

    def reset(self) -> None:
        """Reset the simulator back to its initial start position and zero time."""
        self.position = np.array(self.config.start_pos, dtype=np.float64)
        self.velocity = np.array([self.config.base_speed, 0.0], dtype=np.float64)
        self.frame_count = 0
        self.time = 0.0
