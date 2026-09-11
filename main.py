"""Main entry point and standalone demonstration script for Part 2.

This script coordinates:
1. Beacon ground-truth motion simulation (BeaconSimulator).
2. Optical disturbance, measurement noise, occlusions, and outlier injection (DisturbanceSimulator).
3. Virtual Camera Model (VirtualCamera).
4. Kalman-filter-based predictive tracking and recovery state machine (KalmanBeaconTracker).
5. Closed-loop Camera Pan/Tilt Movement Controller (CameraController).
6. Real-time OpenCV visualization displaying:
   - GREEN: Ground Truth (GT)
   - YELLOW: Optical Measurements (MEAS)
   - RED MARKER: Rejected Outlier Measurements
   - BLUE: Kalman Predicted / Corrected Track
   - SLATE RECTANGLE: Moving Virtual Camera Field of View (FOV)
   - Comprehensive HUD telemetry (State, Confidence, Miss Count, Pan/Tilt Servo Status).

Demonstration Scenarios Covered:
- Scenario A: Initial Pan/Tilt Acquisition & Tracking (camera slews from center (640,360) to beacon (900,450))
- Scenario B: Short detection loss (PREDICTING -> Camera continues predictive pan/tilt servoing)
- Scenario C: Long detection loss (PREDICTING -> LOST -> Camera holds position)
- Scenario D: False/far detection rejection (OUTLIER REJECTED, Camera does not jump)

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
from src.camera_controller import CameraController
from src.kalman_tracker import KalmanBeaconTracker
from src.visualizer import TrackingVisualizer


def run_demonstration(config: SystemConfig = DEFAULT_CONFIG) -> None:
    """Run the live predictive tracking and closed-loop camera control demonstration loop.
    
    Args:
        config: System configuration settings.
    """
    print("=" * 78)
    print("SIH Part 2: Predictive Tracking & Virtual Camera Pan/Tilt Control")
    print("=" * 78)
    print("Scenarios Scheduled:")
    print("  - Frames   0.. 59: Pan/Tilt Slew & Acquisition [TRACKING_SERVO]")
    print("  - Frames  60.. 85: Short occlusion [PREDICTIVE_SERVO -> REACQUIRING]")
    print("  - Frames 160..220: Long occlusion [PREDICTING -> LOST -> HOLDING]")
    print("  - Frames 280..281: Far outlier at (1100, 100) [REJECTED, Camera Stable]")
    print("-" * 78)
    print("Interactive Controls:")
    print("  [SPACE]  - Pause / Resume simulation")
    print("  [F]      - Inject a 60-frame (~2 sec) false outlier at (1100, 100)")
    print("  [R]      - Reset simulation, camera, controller, and tracker to initial state")
    print("  [Q/ESC]  - Quit demonstration")
    print("=" * 78)
    
    beacon_sim = BeaconSimulator(config.beacon)
    disturbance_sim = DisturbanceSimulator(config.disturbance)
    camera = VirtualCamera(config.camera)
    camera_controller = CameraController(camera, config.controller)
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
                
                # 4. Closed-loop Pan/Tilt Camera Controller Step
                # Controller uses ONLY the Kalman tracked / predicted state, never ground truth!
                camera_controller.update(tracking_result, dt)
                
                # 5. Render visual frame (combining GT, Measurement, Kalman Track, Camera FOV, and HUD)
                canvas = visualizer.render_frame(
                    ground_truth=ground_truth_pos,
                    measurement=measurement,
                    tracking_result=tracking_result,
                    frame_idx=frame_idx,
                    fps_display=actual_fps,
                    camera=camera,
                    controller=camera_controller
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
                    camera_controller.reset()
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
                camera_controller.reset()
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
