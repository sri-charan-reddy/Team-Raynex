"""Beacon Motion Model for FSOC Simulation.

This module provides deterministic kinematic trajectory generation (e.g., 1D sinusoidal
or 2D circular) for the remote optical beacon as a function of simulation time.
"""

import math
from typing import Tuple, Dict, Any


class BeaconMotionModel:
    """Calculates dynamic beacon coordinates (x, y) as a function of simulation time.
    
    Attributes:
        motion_type (str): Trajectory type ("sinusoidal", "circular").
        axis (str): Oscillation axis ("y" or "x").
        amplitude (float): Oscillation displacement magnitude in pixels.
        frequency (float): Motion frequency in Hz.
        origin_x (float): Baseline horizontal origin coordinate.
        origin_y (float): Baseline vertical origin coordinate.
        enabled (bool): Master toggle for kinematic motion updates.
    """

    def __init__(
        self,
        motion_type: str = "sinusoidal",
        axis: str = "y",
        amplitude: float = 80.0,
        frequency: float = 0.15,
        origin_x: float = 1100.0,
        origin_y: float = 360.0,
        enabled: bool = False,
    ) -> None:
        self.motion_type = str(motion_type).lower()
        self.axis = str(axis).lower()
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.enabled = bool(enabled)

    def get_position(self, sim_time: float) -> Tuple[float, float]:
        """Computes current (x, y) beacon coordinates at simulation time t.
        
        Args:
            sim_time (float): Elapsed simulation time in seconds.
            
        Returns:
            Tuple[float, float]: Updated (x, y) world coordinates.
        """
        if not self.enabled:
            return (self.origin_x, self.origin_y)

        if self.motion_type == "sinusoidal":
            displacement = self.amplitude * math.sin(2.0 * math.pi * self.frequency * sim_time)
            if self.axis == "x":
                return (self.origin_x + displacement, self.origin_y)
            else:  # Default 'y' axis
                return (self.origin_x, self.origin_y + displacement)
        elif self.motion_type == "circular":
            angle = 2.0 * math.pi * self.frequency * sim_time
            x = self.origin_x + self.amplitude * math.cos(angle)
            y = self.origin_y + self.amplitude * math.sin(angle)
            return (x, y)
        else:
            # Fallback to static origin
            return (self.origin_x, self.origin_y)

    def to_dict(self) -> Dict[str, Any]:
        """Returns motion model parameters as a dictionary."""
        return {
            "motion_type": self.motion_type,
            "axis": self.axis,
            "amplitude": self.amplitude,
            "frequency": self.frequency,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "enabled": self.enabled,
        }
