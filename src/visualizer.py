"""OpenCV-based visualization and manual demonstration tool for Part 2.

This module provides:
- Live 2D canvas rendering of ground-truth beacon motion and noisy optical measurements.
- Visual demonstration of temporary detection loss and occlusions.
- Trajectory history trail rendering for both ground-truth and optical measurements.
- Real-time telemetry Heads-Up Display (HUD) overlay.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import VisualizerConfig
from src.tracking_state import BeaconMeasurement, TrackingResult


class TrackingVisualizer:
    """Renders real-time 2D beacon motion, optical detections, and telemetry HUD."""

    def __init__(self, config: Optional[VisualizerConfig] = None) -> None:
        """Initialize visualization canvas dimensions, color palettes, and trajectory trails.
        
        Args:
            config: Visualizer configuration settings.
        """
        self.config = config or VisualizerConfig()
        self.ground_truth_history: List[Tuple[float, float]] = []
        self.measurement_history: List[Optional[Tuple[float, float]]] = []
        
        # Color palette (BGR format for OpenCV)
        self.COLOR_BG = (22, 24, 28)           # Dark charcoal slate background
        self.COLOR_GRID = (36, 40, 48)         # Subtle background grid
        self.COLOR_GT = (0, 230, 115)          # Bright green for ground truth
        self.COLOR_GT_TRAIL = (0, 140, 70)     # Dimmer green for GT trail
        self.COLOR_MEAS = (0, 215, 255)        # Bright gold / yellow for measurement
        self.COLOR_MEAS_TRAIL = (0, 120, 160)  # Dimmer gold for measurement points
        self.COLOR_TEXT_PRIMARY = (240, 240, 240)
        self.COLOR_TEXT_MUTED = (160, 160, 160)
        self.COLOR_ALERT = (50, 50, 235)       # Vivid red for loss / occlusion alert
        self.COLOR_ACTIVE = (50, 205, 50)      # Vivid green for active detection

    def render_simulation_frame(
        self,
        ground_truth: Tuple[float, float],
        measurement: BeaconMeasurement,
        frame_idx: int,
        fps_display: float = 30.0
    ) -> np.ndarray:
        """Render a single visualization frame for the Phase 2 demonstration.
        
        Args:
            ground_truth: True (x, y) beacon position.
            measurement: BeaconMeasurement containing detection status and measured (x, y).
            frame_idx: Current simulation frame count.
            fps_display: Measured / target display frame rate.
            
        Returns:
            np.ndarray: BGR image canvas suitable for cv2.imshow.
        """
        # 1. Create base canvas with subtle grid
        canvas = np.full(
            (self.config.canvas_height, self.config.canvas_width, 3),
            self.COLOR_BG,
            dtype=np.uint8
        )
        self._draw_grid(canvas)

        # 2. Update trajectory histories
        self.ground_truth_history.append(ground_truth)
        if len(self.ground_truth_history) > self.config.trail_length:
            self.ground_truth_history.pop(0)

        meas_pos = measurement.position if measurement.detected else None
        self.measurement_history.append(meas_pos)
        if len(self.measurement_history) > self.config.trail_length:
            self.measurement_history.pop(0)

        # 3. Draw ground truth trajectory trail
        if self.config.show_ground_truth and len(self.ground_truth_history) >= 2:
            pts = np.array(self.ground_truth_history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=self.COLOR_GT_TRAIL, thickness=2, lineType=cv2.LINE_AA)

        # 4. Draw measurement history dots
        if self.config.show_measurements:
            for pt in self.measurement_history:
                if pt is not None:
                    cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, self.COLOR_MEAS_TRAIL, -1, lineType=cv2.LINE_AA)

        # 5. Draw current ground-truth beacon (Green circle + crosshair)
        if self.config.show_ground_truth:
            gt_x, gt_y = int(ground_truth[0]), int(ground_truth[1])
            cv2.circle(canvas, (gt_x, gt_y), 10, self.COLOR_GT, 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (gt_x, gt_y), 3, self.COLOR_GT, -1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, "GT", (gt_x + 14, gt_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_GT, 1, cv2.LINE_AA)

        # 6. Draw current measurement (Yellow circle + target lines) ONLY if detected
        if measurement.detected and measurement.position is not None:
            m_x, m_y = int(measurement.position[0]), int(measurement.position[1])
            cv2.circle(canvas, (m_x, m_y), 14, self.COLOR_MEAS, 2, lineType=cv2.LINE_AA)
            cv2.drawMarker(canvas, (m_x, m_y), self.COLOR_MEAS, markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1, line_type=cv2.LINE_AA)
            cv2.putText(canvas, "MEAS", (m_x + 18, m_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_MEAS, 1, cv2.LINE_AA)

        # 7. Draw telemetry HUD panel
        self._draw_hud(canvas, ground_truth, measurement, frame_idx, fps_display)

        return canvas

    def _draw_grid(self, canvas: np.ndarray, step: int = 80) -> None:
        """Draw subtle Cartesian grid lines across the canvas."""
        h, w = canvas.shape[:2]
        for x in range(0, w, step):
            cv2.line(canvas, (x, 0), (x, h), self.COLOR_GRID, 1)
        for y in range(0, h, step):
            cv2.line(canvas, (0, y), (w, y), self.COLOR_GRID, 1)

    def _draw_hud(
        self,
        canvas: np.ndarray,
        ground_truth: Tuple[float, float],
        measurement: BeaconMeasurement,
        frame_idx: int,
        fps: float
    ) -> None:
        """Render top-left telemetry HUD and status banner."""
        # Top-left HUD card background
        card_w, card_h = 420, 190
        sub_img = canvas[15:15 + card_h, 15:15 + card_w]
        white_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.35, white_rect, 0.65, 1.0)
        canvas[15:15 + card_h, 15:15 + card_w] = res
        cv2.rectangle(canvas, (15, 15), (15 + card_w, 15 + card_h), (60, 68, 80), 1)

        # Header
        cv2.putText(canvas, "SIH PART 2 - MOTION & DISTURBANCE SIM (PHASE 2)", (28, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (28, 48), (28 + card_w - 26, 48), (60, 68, 80), 1)

        # Telemetry fields
        line_y = 70
        spacing = 24

        # Frame counter & FPS
        cv2.putText(canvas, f"Frame: {frame_idx:04d}  |  FPS: {fps:.1f}", (28, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
        
        # Detection Status
        line_y += spacing
        if measurement.detected:
            det_text = "Detection: YES (Optical Lock Active)"
            det_color = self.COLOR_ACTIVE
        else:
            det_text = "Detection: NO (OCCLUDED / LOST)"
            det_color = self.COLOR_ALERT

        cv2.putText(canvas, det_text, (28, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 2, cv2.LINE_AA)

        # Ground truth coordinates
        line_y += spacing
        gt_str = f"Ground Truth: ({ground_truth[0]:.1f}, {ground_truth[1]:.1f})"
        cv2.putText(canvas, gt_str, (28, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_GT, 1, cv2.LINE_AA)

        # Measurement coordinates
        line_y += spacing
        if measurement.detected and measurement.position is not None:
            meas_str = f"Measurement:  ({measurement.position[0]:.1f}, {measurement.position[1]:.1f})"
            meas_color = self.COLOR_MEAS
        else:
            meas_str = "Measurement:  NONE (Signal Occluded)"
            meas_color = self.COLOR_ALERT

        cv2.putText(canvas, meas_str, (28, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, meas_color, 1, cv2.LINE_AA)

        # Controls hint
        line_y += spacing
        cv2.putText(canvas, "[SPACE] Pause/Play  |  [R] Reset  |  [Q/ESC] Quit", (28, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Center top banner during occlusion
        if not measurement.detected:
            banner_text = "--- TEMPORARY OPTICAL OCCLUSION ACTIVE ---"
            text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            b_x = (self.config.canvas_width - text_size[0]) // 2
            cv2.rectangle(canvas, (b_x - 15, 20), (b_x + text_size[0] + 15, 55), (0, 0, 150), -1)
            cv2.putText(canvas, banner_text, (b_x, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def reset(self) -> None:
        """Clear all historical trajectory trails."""
        self.ground_truth_history.clear()
        self.measurement_history.clear()
