"""Main entry point and standalone demonstration script for Part 2 (Phase 2).

This script coordinates:
1. Beacon ground-truth motion simulation (BeaconSimulator).
2. Optical disturbance, measurement noise, and occlusion generation (DisturbanceSimulator).
3. Real-time OpenCV visualization displaying moving ground-truth, noisy optical measurements,
   and visual detection-loss states.

Usage:
    python main.py
"""

import sys
import time
import cv2

from config import DEFAULT_CONFIG, SystemConfig
from src.beacon_simulator import BeaconSimulator
from src.disturbance_simulator import DisturbanceSimulator
from src.visualizer import TrackingVisualizer


def run_demonstration(config: SystemConfig = DEFAULT_CONFIG) -> None:
    """Run the live beacon motion and disturbance simulation loop.
    
    Args:
        config: System configuration settings.
    """
    print("=" * 68)
    print("SIH Part 2: Beacon Motion & Disturbance Simulation (Phase 2)")
    print("=" * 68)
    print("Controls:")
    print("  [SPACE] - Pause / Resume simulation")
    print("  [R]     - Reset simulation to initial state")
    print("  [Q/ESC] - Quit demonstration")
    print("=" * 68)
    
    beacon_sim = BeaconSimulator(config.beacon)
    disturbance_sim = DisturbanceSimulator(config.disturbance)
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
                
                # 1. Step ground-truth beacon motion
                ground_truth_pos = beacon_sim.step(dt)
                
                # 2. Apply disturbances, sensor noise, and scheduled occlusions
                measurement = disturbance_sim.apply_disturbances(ground_truth_pos, timestamp, frame_idx)
                
                # Conceptual Phase 2 clean data interface:
                # measured_position: Optional[Tuple[float, float]] = measurement.position if measurement.detected else None
                # detection_available: bool = measurement.detected
                
                # 3. Render visualization frame
                canvas = visualizer.render_simulation_frame(
                    ground_truth=ground_truth_pos,
                    measurement=measurement,
                    frame_idx=frame_idx,
                    fps_display=actual_fps
                )
                
                cv2.imshow(window_name, canvas)
                frame_idx += 1
            else:
                # If paused, keep displaying the current canvas
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key == ord(' '):
                    paused = False
                elif key in (ord('r'), ord('R')):
                    beacon_sim.reset()
                    disturbance_sim.reset()
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
            
            # Maintain nominal frame rate
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
