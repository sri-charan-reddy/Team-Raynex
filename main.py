"""Main entry point and standalone demonstration script for Part 2.

This script coordinates:
1. Beacon ground-truth motion simulation (BeaconSimulator).
2. Optical disturbance, measurement noise, occlusions, and outlier injection (DisturbanceSimulator).
3. Virtual Camera Model with configurable Field of View (VirtualCamera).
4. Kalman-filter-based predictive tracking with outlier rejection and recovery state machine (KalmanBeaconTracker).
5. Real-time OpenCV visualization displaying:
   - GREEN: Ground Truth (GT)
   - YELLOW: Optical Measurements (MEAS)
   - RED MARKER: Rejected Outlier Measurements
   - BLUE: Kalman Predicted / Corrected Track
   - SLATE RECTANGLE: Virtual Camera Field of View (FOV)
   - Comprehensive HUD telemetry (State, Confidence, Miss Count, Camera FOV Telemetry).

Demonstration Scenarios Covered:
- Scenario A: Normal tracking (TRACKING)
- Scenario B: Short detection loss (PREDICTING -> REACQUIRING -> TRACKING)
- Scenario C: Long detection loss (PREDICTING -> LOST -> REACQUIRING -> TRACKING)
- Scenario D: False/far detection rejection (OUTLIER REJECTED, Kalman track remains stable)

Usage:
    python main.py
"""

import sys
import time
import cv2

from config import DEFAULT_CONFIG, SystemConfig
from src.beacon_simulator import BeaconSimulator
from src.disturbance_simulator import DisturbanceSimulator
from src.virtual_camera import VirtualCamera
from src.kalman_tracker import KalmanBeaconTracker
from src.visualizer import TrackingVisualizer


def run_demonstration(config: SystemConfig = DEFAULT_CONFIG) -> None:
    """Run the live robust predictive tracking and virtual camera demonstration loop.
    
    Args:
        config: System configuration settings.
    """
    print("=" * 76)
    print("SIH Part 2: Robust Predictive Tracking & Virtual Camera Model")
    print("=" * 76)
    print("Scenarios Scheduled:")
    print("  - Frames   0.. 59: Normal tracking [TRACKING]")
    print("  - Frames  60.. 85: Short occlusion [PREDICTING -> REACQUIRING -> TRACKING]")
    print("  - Frames 150..210: Long occlusion [PREDICTING -> LOST -> REACQUIRING -> TRACKING]")
    print("  - Frames 270..271: Far outlier injection at (1100, 100) [OUTLIER REJECTED]")
    print("-" * 76)
    print("Interactive Controls:")
    print("  [SPACE]  - Pause / Resume simulation")
    print("  [F]      - Inject a 60-frame (~2 sec) false outlier at (1100, 100)")
    print("  [R]      - Reset simulation, camera, and tracker to initial state")
    print("  [Q/ESC]  - Quit demonstration")
    print("=" * 76)
    
    beacon_sim = BeaconSimulator(config.beacon)
    disturbance_sim = DisturbanceSimulator(config.disturbance)
    camera = VirtualCamera(config.camera)
    tracker = KalmanBeaconTracker(config.kalman, config.tracking)
    visualizer = TrackingVisualizer(config.visualizer)
    
    window_name = config.visualizer.window_name
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    frame_idx = 0
    fps = config.visualizer.fps
    dt = 1.0 / max(1, fps)
    paused = False
    
    prev_time = time.time()
    actual_fps = float(fps)
    
    try:
        while True:
            loop_start = time.time()
            
            if not paused:
                timestamp = frame_idx * dt
                
                # 1. Ground truth beacon motion simulation
                ground_truth_pos = beacon_sim.step(dt)
                
                # 2. Disturbance simulation (noise, scheduled occlusions, scheduled/manual outliers)
                measurement = disturbance_sim.apply_disturbances(ground_truth_pos, timestamp, frame_idx)
                
                # 3. Kalman Filter with Outlier Rejection & Recovery State Machine
                # The tracker ONLY receives the optical measurement.
                # Ground truth is never accessed by the tracker.
                tracking_result = tracker.process_frame(measurement, timestamp)
                
                # 4. Render visual frame (including Virtual Camera FOV)
                canvas = visualizer.render_frame(
                    ground_truth=ground_truth_pos,
                    measurement=measurement,
                    tracking_result=tracking_result,
                    frame_idx=frame_idx,
                    fps_display=actual_fps,
                    camera=camera
                )
                
                cv2.imshow(window_name, canvas)
                frame_idx += 1
            else:
                # Paused loop
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key == ord(' '):
                    paused = False
                elif key in (ord('f'), ord('F')):
                    disturbance_sim.trigger_manual_outlier((1100.0, 100.0), duration_frames=60)
                    print(f"[Frame {frame_idx}] Manual outlier triggered at (1100.0, 100.0) for 60 frames (~2 sec).")
                elif key in (ord('r'), ord('R')):
                    beacon_sim.reset()
                    disturbance_sim.reset()
                    camera.reset()
                    tracker.reset()
                    visualizer.reset()
                    frame_idx = 0
                    paused = False
                continue
            
            # Telemetry FPS calculation
            curr_time = time.time()
            elapsed = curr_time - prev_time
            if elapsed > 0:
                actual_fps = 0.9 * actual_fps + 0.1 * (1.0 / elapsed)
            prev_time = curr_time
            
            # Frame timing
            compute_time = curr_time - loop_start
            wait_ms = max(1, int((dt - compute_time) * 1000))
            
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("Exit requested by user.")
                break
            elif key == ord(' '):
                paused = True
            elif key in (ord('f'), ord('F')):
                disturbance_sim.trigger_manual_outlier((1100.0, 100.0), duration_frames=60)
                print(f"[Frame {frame_idx}] Manual outlier triggered at (1100.0, 100.0) for 60 frames (~2 sec).")
            elif key in (ord('r'), ord('R')):
                beacon_sim.reset()
                disturbance_sim.reset()
                camera.reset()
                tracker.reset()
                visualizer.reset()
                frame_idx = 0

    finally:
        cv2.destroyAllWindows()
        print("Demonstration ended.")


def main() -> None:
    """CLI entry point."""
    try:
        run_demonstration()
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
