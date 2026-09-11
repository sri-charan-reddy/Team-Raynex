"""OpenCV-based visualization and manual demonstration tool for Part 2 (Phase 3).

This module provides:
- Live 2D canvas rendering of:
  1. Ground-Truth motion trajectory (GREEN)
  2. Actual optical measurements with sensor noise (YELLOW)
  3. Kalman predicted / corrected tracking trajectory (BLUE)
- Visual demonstration of predictive coasting during optical occlusion.
- Real-time telemetry Heads-Up Display (HUD) indicating Detection status, Mode, Coordinates, and Velocity.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import VisualizerConfig
from src.tracking_state import BeaconMeasurement, TrackingResult, TrackingState


class TrackingVisualizer:
    """Renders real-time 2D beacon motion, optical detections, Kalman tracks, and telemetry HUD."""

    def __init__(self, config: Optional[VisualizerConfig] = None) -> None:
        """Initialize visualization canvas dimensions, color palettes, and trajectory trails.
        
        Args:
            config: Visualizer configuration settings.
        """
        self.config = config or VisualizerConfig()
        self.ground_truth_history: List[Tuple[float, float]] = []
        self.measurement_history: List[Optional[Tuple[float, float]]] = []
        self.kalman_history: List[Tuple[float, float]] = []
        
        # Color palette (BGR format for OpenCV)
        self.COLOR_BG = (22, 24, 28)             # Dark slate background
        self.COLOR_GRID = (36, 40, 48)           # Background grid
        self.COLOR_GT = (0, 230, 115)            # GREEN for ground truth
        self.COLOR_GT_TRAIL = (0, 140, 70)       # Dimmer green trail
        self.COLOR_MEAS = (0, 215, 255)          # YELLOW / GOLD for optical measurements
        self.COLOR_MEAS_TRAIL = (0, 120, 160)    # Dimmer gold points
        self.COLOR_KALMAN = (255, 175, 40)       # BLUE / CYAN for Kalman track (BGR: 255, 175, 40 is rich bright electric blue/sky-blue)
        self.COLOR_KALMAN_TRAIL = (180, 110, 20) # Dimmer blue trail
        self.COLOR_TEXT_PRIMARY = (240, 240, 240)
        self.COLOR_TEXT_MUTED = (160, 160, 160)
        self.COLOR_ALERT = (50, 50, 235)         # RED for occlusion alert
        self.COLOR_ACTIVE = (50, 205, 50)        # GREEN for active detection
        self.COLOR_PRED_BADGE = (255, 165, 0)    # Orange/Cyan for Predicting badge

    def render_frame(
        self,
        ground_truth: Tuple[float, float],
        measurement: BeaconMeasurement,
        tracking_result: TrackingResult,
        frame_idx: int,
        fps_display: float = 30.0
    ) -> np.ndarray:
        """Render a single visualization frame combining GT, Measurement, Kalman track, and HUD.
        
        Args:
            ground_truth: True (x, y) beacon position.
            measurement: BeaconMeasurement containing detection status and measured (x, y).
            tracking_result: TrackingResult containing Kalman predicted/corrected state.
            frame_idx: Current simulation frame count.
            fps_display: Live measured FPS.
            
        Returns:
            np.ndarray: BGR image canvas suitable for cv2.imshow.
        """
        # 1. Create base canvas with grid
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

        kalman_pos = tracking_result.position
        self.kalman_history.append(kalman_pos)
        if len(self.kalman_history) > self.config.trail_length:
            self.kalman_history.pop(0)

        # 3. Draw Ground Truth Trajectory Trail (GREEN)
        if self.config.show_ground_truth and len(self.ground_truth_history) >= 2:
            pts = np.array(self.ground_truth_history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=self.COLOR_GT_TRAIL, thickness=2, lineType=cv2.LINE_AA)

        # 4. Draw Measurement History Dots (YELLOW)
        if self.config.show_measurements:
            for pt in self.measurement_history:
                if pt is not None:
                    cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, self.COLOR_MEAS_TRAIL, -1, lineType=cv2.LINE_AA)

        # 5. Draw Kalman Track Trajectory Trail (BLUE)
        if len(self.kalman_history) >= 2:
            k_pts = np.array(self.kalman_history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [k_pts], isClosed=False, color=self.COLOR_KALMAN_TRAIL, thickness=2, lineType=cv2.LINE_AA)

        # 6. Draw Ground Truth Beacon Position (GREEN circle)
        if self.config.show_ground_truth:
            gt_x, gt_y = int(ground_truth[0]), int(ground_truth[1])
            cv2.circle(canvas, (gt_x, gt_y), 9, self.COLOR_GT, 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (gt_x, gt_y), 3, self.COLOR_GT, -1, lineType=cv2.LINE_AA)
            cv2.putText(canvas, "GT", (gt_x + 12, gt_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_GT, 1, cv2.LINE_AA)

        # 7. Draw Optical Measurement (YELLOW crosshair) ONLY if detected
        if measurement.detected and measurement.position is not None:
            m_x, m_y = int(measurement.position[0]), int(measurement.position[1])
            cv2.circle(canvas, (m_x, m_y), 13, self.COLOR_MEAS, 1, lineType=cv2.LINE_AA)
            cv2.drawMarker(canvas, (m_x, m_y), self.COLOR_MEAS, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1, line_type=cv2.LINE_AA)
            cv2.putText(canvas, "MEAS", (m_x + 16, m_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_MEAS, 1, cv2.LINE_AA)

        # 8. Draw Kalman Track Position (BLUE circle / target)
        if tracking_result.state != TrackingState.UNINITIALIZED:
            k_x, k_y = int(kalman_pos[0]), int(kalman_pos[1])
            cv2.circle(canvas, (k_x, k_y), 16, self.COLOR_KALMAN, 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (k_x, k_y), 3, self.COLOR_KALMAN, -1, lineType=cv2.LINE_AA)
            
            # Label Kalman position
            mode_lbl = "KALMAN (PRED)" if tracking_result.is_predicted else "KALMAN (TRACK)"
            cv2.putText(canvas, mode_lbl, (k_x + 18, k_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_KALMAN, 1, cv2.LINE_AA)

        # 9. Draw Telemetry HUD
        self._draw_hud(canvas, ground_truth, measurement, tracking_result, frame_idx, fps_display)

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
        tracking_result: TrackingResult,
        frame_idx: int,
        fps: float
    ) -> None:
        """Render top-left telemetry HUD card and status alert banner."""
        card_w, card_h = 440, 230
        sub_img = canvas[15:15 + card_h, 15:15 + card_w]
        white_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.35, white_rect, 0.65, 1.0)
        canvas[15:15 + card_h, 15:15 + card_w] = res
        cv2.rectangle(canvas, (15, 15), (15 + card_w, 15 + card_h), (60, 68, 80), 1)

        # Header
        cv2.putText(canvas, "SIH PART 2 - KALMAN PREDICTIVE TRACKING", (26, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 200, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (26, 46), (26 + card_w - 24, 46), (60, 68, 80), 1)

        line_y = 66
        spacing = 22

        # 1. Frame counter & FPS
        cv2.putText(canvas, f"Frame: {frame_idx:04d}  |  FPS: {fps:.1f}", (26, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
        
        # 2. Detection Status
        line_y += spacing
        if measurement.detected:
            det_text = "Detection: YES (Optical Measurement Available)"
            det_color = self.COLOR_ACTIVE
        else:
            det_text = "Detection: NO (Optical Occlusion Active)"
            det_color = self.COLOR_ALERT

        cv2.putText(canvas, det_text, (26, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, det_color, 1, cv2.LINE_AA)

        # 3. Ground Truth Position
        line_y += spacing
        gt_str = f"Ground Truth: ({ground_truth[0]:.1f}, {ground_truth[1]:.1f})"
        cv2.putText(canvas, gt_str, (26, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.COLOR_GT, 1, cv2.LINE_AA)

        # 4. Measurement Position
        line_y += spacing
        if measurement.detected and measurement.position is not None:
            meas_str = f"Measurement:  ({measurement.position[0]:.1f}, {measurement.position[1]:.1f})"
            meas_color = self.COLOR_MEAS
        else:
            meas_str = "Measurement:  NONE"
            meas_color = self.COLOR_ALERT

        cv2.putText(canvas, meas_str, (26, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, meas_color, 1, cv2.LINE_AA)

        # 5. Kalman Track Position
        line_y += spacing
        k_str = f"Kalman Track: ({tracking_result.position[0]:.1f}, {tracking_result.position[1]:.1f})"
        cv2.putText(canvas, k_str, (26, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.COLOR_KALMAN, 1, cv2.LINE_AA)

        # 6. Velocity Vector
        line_y += spacing
        vel_str = f"Velocity:     ({tracking_result.velocity[0]:.1f}, {tracking_result.velocity[1]:.1f}) px/s"
        cv2.putText(canvas, vel_str, (26, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)

        # 7. Mode (TRACKING vs PREDICTING)
        line_y += spacing
        if tracking_result.is_predicted:
            mode_str = f"Mode: PREDICTING (Coast Frame {tracking_result.frames_without_detection})"
            mode_color = (0, 165, 255) # Bright orange
        else:
            mode_str = "Mode: TRACKING (Kalman Corrected)"
            mode_color = self.COLOR_ACTIVE

        cv2.putText(canvas, mode_str, (26, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, mode_color, 2, cv2.LINE_AA)

        # Center top banner during occlusion
        if not measurement.detected:
            banner_text = "--- OCCLUSION: KALMAN DEAD RECKONING / PREDICTING ---"
            text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            b_x = (self.config.canvas_width - text_size[0]) // 2
            cv2.rectangle(canvas, (b_x - 14, 18), (b_x + text_size[0] + 14, 52), (0, 0, 150), -1)
            cv2.putText(canvas, banner_text, (b_x, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    def reset(self) -> None:
        """Clear all historical trajectory trails."""
        self.ground_truth_history.clear()
        self.measurement_history.clear()
        self.kalman_history.clear()
