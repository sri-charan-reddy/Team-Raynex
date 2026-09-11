"""Main entry point and standalone demonstration script for Part 2.

This script coordinates:
1. Beacon ground-truth motion simulation (BeaconSimulator).
2. Optical disturbance, measurement noise, occlusions, and outlier injection (DisturbanceSimulator).
3. Virtual Camera Model (VirtualCamera).
4. Kalman-filter-based predictive tracking and recovery state machine (KalmanBeaconTracker).
5. Bounded Local Search around Kalman prediction (LocalSearch).
6. Closed-loop Camera Pan/Tilt Movement Controller (CameraController).
7. Real-time OpenCV visualization displaying:
   - GREEN: Ground Truth (GT)
   - YELLOW: Optical Measurements (MEAS)
   - RED MARKER: Rejected Outlier Measurements
   - BLUE: Kalman Predicted / Corrected Track
   - SLATE RECTANGLE: Moving Virtual Camera Field of View (FOV)
   - CYAN/GOLD CIRCLE & WAYPOINTS: Bounded Local Search Scan Pattern
   - Comprehensive HUD telemetry (State, Confidence, Miss Count, Local Search & Pan/Tilt Status).

Demonstration Scenarios Covered:
- Scenario A: Initial Pan/Tilt Acquisition & Tracking (camera slews from center (640,360) to beacon (900,450))
- Scenario B: Short detection loss (PREDICTING -> Camera continues predictive pan/tilt servoing)
- Scenario C: Extended detection loss (PREDICTING -> SEARCHING: Local search scans bounded region around Kalman prediction -> REACQUIRING / LOST)
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
from src.local_search import LocalSearch
from src.kalman_tracker import KalmanBeaconTracker
from src.tracking_state import TrackingState, PerformanceTracker
from src.visualizer import TrackingVisualizer


def run_demonstration(config: SystemConfig = DEFAULT_CONFIG) -> None:
    """Run the live predictive tracking, local search, turbulence, and camera control demonstration loop.
    
    Args:
        config: System configuration settings.
    """
    print("=" * 78)
    print("SIH Part 2: Predictive Tracking, Turbulence Simulation & Camera Control")
    print("=" * 78)
    print("Scenarios Scheduled:")
    print("  - Frames   0.. 59: Pan/Tilt Slew & Acquisition [TRACKING_SERVO]")
    print("  - Frames  60.. 80: Short occlusion [PREDICTIVE_SERVO -> REACQUIRING]")
    print("  - Frames 150..220: Extended occlusion [PREDICTING -> SEARCHING (Local Search) -> LOST/REACQ]")
    print("  - Frames 290..291: Far outlier at (1100, 100) [REJECTED, Camera Stable]")
    print("-" * 78)
    print("Interactive Controls:")
    print("  [SPACE]  - Pause / Resume simulation")
    print("  [T]      - Toggle / Cycle Atmospheric Turbulence (OFF -> LOW -> MED -> HIGH)")
    print("  [F]      - Inject a 60-frame (~2 sec) false outlier at (1100, 100)")
    print("  [R]      - Reset simulation, camera, search, and tracker to initial state")
    print("  [Q/ESC]  - Quit demonstration")
    print("=" * 78)
    
    beacon_sim = BeaconSimulator(config.beacon)
    disturbance_sim = DisturbanceSimulator(config.disturbance)
    camera = VirtualCamera(config.camera)
    local_search = LocalSearch(config.search)
    camera_controller = CameraController(camera, config.controller)
    tracker = KalmanBeaconTracker(config.kalman, config.tracking)
    perf_tracker = PerformanceTracker()
    visualizer = TrackingVisualizer(config.visualizer)
    
    window_name = config.visualizer.window_name
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    frame_idx = 0
    fps = config.visualizer.fps
    dt = 1.0 / max(1, fps)
    paused = False
    
    prev_time = time.time()
    actual_fps = float(fps)
    last_metrics = None
    
    try:
        while True:
            loop_start = time.time()
            
            if not paused:
                timestamp = frame_idx * dt
                
                # 1. Ground truth beacon motion simulation
                ground_truth_pos = beacon_sim.step(dt)
                
                # 2. Disturbance simulation (atmospheric turbulence, noise, scheduled occlusions, outliers)
                measurement = disturbance_sim.apply_disturbances(ground_truth_pos, timestamp, frame_idx)
                
                # 3. Kalman Filter with Outlier Rejection & State Machine (PREDICTING -> SEARCHING -> LOST)
                # Tracker receives strictly optical detection data; ground truth is completely hidden.
                tracking_result = tracker.process_frame(measurement, timestamp)
                
                # 4. Local Search Waypoint Generation
                # Operates around the CURRENT KALMAN PREDICTED POSITION (never ground truth)
                is_searching = (tracking_result.state == TrackingState.SEARCHING)
                search_target = local_search.update(tracking_result.position, is_searching)
                
                # 5. Closed-loop Pan/Tilt Camera Controller Step
                # Steers camera toward search target during SEARCHING, or Kalman track during TRACKING/PREDICTING
                camera_controller.update(
                    tracking_result,
                    dt,
                    target_override=search_target if is_searching else None
                )
                
                # Record loop compute time
                compute_time_s = time.time() - loop_start
                
                # 6. Update Performance Metrics
                metrics = perf_tracker.update(
                    ground_truth=ground_truth_pos,
                    measurement=measurement,
                    tracking_result=tracking_result,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    frame_compute_time_s=compute_time_s
                )
                last_metrics = metrics
                
                # 7. Render visual frame
                canvas = visualizer.render_frame(
                    ground_truth=ground_truth_pos,
                    measurement=measurement,
                    tracking_result=tracking_result,
                    frame_idx=frame_idx,
                    fps_display=actual_fps,
                    camera=camera,
                    controller=camera_controller,
                    search=local_search,
                    turbulence=disturbance_sim.turbulence,
                    metrics=metrics
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
                elif key in (ord('t'), ord('T')):
                    new_lvl = disturbance_sim.turbulence.toggle_level()
                    print(f"[Frame {frame_idx}] Atmospheric turbulence level changed to: {new_lvl.value}")
                elif key in (ord('f'), ord('F')):
                    disturbance_sim.trigger_manual_outlier((1100.0, 100.0), duration_frames=60)
                    print(f"[Frame {frame_idx}] Manual outlier triggered at (1100.0, 100.0) for 60 frames (~2 sec).")
                elif key in (ord('r'), ord('R')):
                    beacon_sim.reset()
                    disturbance_sim.reset()
                    camera.reset()
                    local_search.reset()
                    camera_controller.reset()
                    tracker.reset()
                    perf_tracker.reset()
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
            elif key in (ord('t'), ord('T')):
                new_lvl = disturbance_sim.turbulence.toggle_level()
                print(f"[Frame {frame_idx}] Atmospheric turbulence level changed to: {new_lvl.value}")
            elif key in (ord('f'), ord('F')):
                disturbance_sim.trigger_manual_outlier((1100.0, 100.0), duration_frames=60)
                print(f"[Frame {frame_idx}] Manual outlier triggered at (1100.0, 100.0) for 60 frames (~2 sec).")
            elif key in (ord('r'), ord('R')):
                beacon_sim.reset()
                disturbance_sim.reset()
                camera.reset()
                local_search.reset()
                camera_controller.reset()
                tracker.reset()
                perf_tracker.reset()
                visualizer.reset()
                frame_idx = 0

    finally:
        cv2.destroyAllWindows()
        print("\n" + "=" * 78)
        print("SIH PART 2: FINAL TRACKING PERFORMANCE SUMMARY")
        print("=" * 78)
        if last_metrics is not None:
            print(f"  Simulation Duration:       {last_metrics.simulation_duration:.2f} s ({last_metrics.total_frames} frames)")
            print(f"  Average Frame Rate:        {last_metrics.avg_fps:.1f} FPS")
            print(f"  Initial Lock Time:         {last_metrics.acquisition_time_s:.2f} s (Frame #{last_metrics.acquisition_frame})")
            print(f"  Average Tracking Error:    {last_metrics.avg_tracking_error_px:.2f} px")
            print(f"  Maximum Tracking Error:    {last_metrics.max_tracking_error_px:.2f} px")
            print(f"  Track Lock Retention:      {last_metrics.lock_retention_pct:.1f}%")
            print(f"  Average Processing Time:   {last_metrics.avg_processing_time_ms:.3f} ms / frame")
            print(f"  Temporary Loss Intervals:  {last_metrics.occlusion_count}")
            print(f"  Successful Reacquisitions: {last_metrics.reacquisition_count}")
            print(f"  Rejected False Outliers:   {last_metrics.rejected_outlier_count}")
        print("=" * 78)
        print("Demonstration ended successfully.")


def main() -> None:
    """CLI entry point."""
    try:
        run_demonstration()
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
