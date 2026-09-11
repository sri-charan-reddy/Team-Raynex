"""Main entry point and standalone demonstration script for Part 2 (Phase 3).

This script coordinates:
1. Beacon ground-truth motion simulation (BeaconSimulator).
2. Optical disturbance, measurement noise, and occlusion generation (DisturbanceSimulator).
3. Kalman-filter-based predictive tracking and state estimation (KalmanBeaconTracker).
4. Real-time OpenCV visualization displaying:
   - GREEN: Ground Truth
   - YELLOW: Optical Measurements
   - BLUE: Kalman Predicted / Corrected Track

Usage:
    python main.py
"""

import sys
import time
import cv2

from config import DEFAULT_CONFIG, SystemConfig
from src.beacon_simulator import BeaconSimulator
from src.disturbance_simulator import DisturbanceSimulator
from src.kalman_tracker import KalmanBeaconTracker
from src.visualizer import TrackingVisualizer


def run_demonstration(config: SystemConfig = DEFAULT_CONFIG) -> None:
    """Run the live Kalman predictive tracking demonstration loop.
    
    Args:
        config: System configuration settings.
    """
    print("=" * 72)
    print("SIH Part 2: Kalman Predictive Tracking & Disturbance Handling (Phase 3)")
    print("=" * 72)
    print("Color Legend:")
    print("  [GREEN]  - Ground Truth trajectory (GT)")
    print("  [YELLOW] - Noisy optical measurements (MEAS) - disappears during occlusion")
    print("  [BLUE]   - Kalman filter track (predicts continuously through occlusions)")
    print("-" * 72)
    print("Controls:")
    print("  [SPACE]  - Pause / Resume simulation")
    print("  [R]      - Reset simulation and tracker to initial state")
    print("  [Q/ESC]  - Quit demonstration")
    print("=" * 72)
    
    beacon_sim = BeaconSimulator(config.beacon)
    disturbance_sim = DisturbanceSimulator(config.disturbance)
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
                
                # 2. Optical disturbance, measurement noise, and occlusion simulation
                measurement = disturbance_sim.apply_disturbances(ground_truth_pos, timestamp, frame_idx)
                
                # 3. Kalman Filter Predictive Tracking
                # Note: Tracker ONLY receives the optical measurement (or None when occluded).
                # Tracker has NO access to ground truth.
                tracking_result = tracker.process_frame(measurement, timestamp)
                
                # 4. Render visualization combining GT, Measurement, and Kalman Track
                canvas = visualizer.render_frame(
                    ground_truth=ground_truth_pos,
                    measurement=measurement,
                    tracking_result=tracking_result,
                    frame_idx=frame_idx,
                    fps_display=actual_fps
                )
                
                cv2.imshow(window_name, canvas)
                frame_idx += 1
            else:
                # If paused, wait for key input
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key == ord(' '):
                    paused = False
                elif key in (ord('r'), ord('R')):
                    beacon_sim.reset()
                    disturbance_sim.reset()
                    tracker.reset()
                    visualizer.reset()
                    frame_idx = 0
                    paused = False
                continue
            
            # Compute FPS telemetry
            curr_time = time.time()
            elapsed = curr_time - prev_time
            if elapsed > 0:
                actual_fps = 0.9 * actual_fps + 0.1 * (1.0 / elapsed)
            prev_time = curr_time
            
            # Frame rate timing control
            compute_time = curr_time - loop_start
            wait_ms = max(1, int((dt - compute_time) * 1000))
            
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("Exit requested by user.")
                break
            elif key == ord(' '):
                paused = True
            elif key in (ord('r'), ord('R')):
                beacon_sim.reset()
                disturbance_sim.reset()
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
