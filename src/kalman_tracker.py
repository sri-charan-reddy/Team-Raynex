"""Kalman-filter-based predictive tracker for moving optical beacons.

This module provides:
- 2D Linear Kalman Filter implementation using OpenCV (cv2.KalmanFilter).
- Kinematic constant-velocity state model [x, y, vx, vy]^T.
- Extrapolation / coasting during optical occlusion (prediction without measurement).
- Seamless correction and state convergence upon measurement return / reacquisition.
- Tracking state machine and confidence reporting for downstream controller integration.
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np

from config import KalmanConfig, TrackingStateConfig
from src.tracking_state import BeaconMeasurement, TrackingResult, TrackingState


class KalmanBeaconTracker:
    """Predictive Kalman Filter Tracker for moving optical beacon localization.
    
    State Vector (4x1):
        x = [x, y, vx, vy]^T
        x, y: 2D position in pixel coordinates
        vx, vy: 2D velocity in pixels per second
    
    Measurement Vector (2x1):
        z = [x_meas, y_meas]^T
    """

    def __init__(
        self,
        kalman_cfg: Optional[KalmanConfig] = None,
        state_cfg: Optional[TrackingStateConfig] = None
    ) -> None:
        """Initialize the OpenCV Kalman filter matrices and tracking state machine.
        
        Args:
            kalman_cfg: Kalman filter tuning parameters.
            state_cfg: Tracking state machine configuration parameters.
        """
        self.kalman_cfg = kalman_cfg or KalmanConfig()
        self.state_cfg = state_cfg or TrackingStateConfig()
        
        self.state: TrackingState = TrackingState.UNINITIALIZED
        self.initialized: bool = False
        self.confidence: float = 0.0
        self.frames_without_detection: int = 0
        self.consecutive_detections: int = 0
        self.last_measurement: Optional[Tuple[float, float]] = None
        
        # Internal state buffer (4x1)
        self.current_state: np.ndarray = np.zeros((4, 1), dtype=np.float32)
        
        # Instantiate OpenCV Kalman Filter (4 dynamic params, 2 measurement params)
        self.kf = cv2.KalmanFilter(4, 2, 0)
        self._setup_kalman_matrices()

    def _setup_kalman_matrices(self) -> None:
        """Configure OpenCV Kalman filter matrices from configuration parameters."""
        dt = float(self.kalman_cfg.dt)
        
        # 1. State Transition Matrix F (Constant Velocity Model)
        # x_k = x_{k-1} + vx_{k-1} * dt
        # y_k = y_{k-1} + vy_{k-1} * dt
        # vx_k = vx_{k-1}
        # vy_k = vy_{k-1}
        self.kf.transitionMatrix = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # 2. Measurement Matrix H (we directly observe position x, y)
        self.kf.measurementMatrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        # 3. Process Noise Covariance Matrix Q
        q_pos = float(self.kalman_cfg.process_noise_std_pos ** 2)
        q_vel = float(self.kalman_cfg.process_noise_std_vel ** 2)
        self.kf.processNoiseCov = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)

        # 4. Measurement Noise Covariance Matrix R
        r_meas = float(self.kalman_cfg.measurement_noise_std ** 2)
        self.kf.measurementNoiseCov = np.diag([r_meas, r_meas]).astype(np.float32)

        # 5. Initial Estimation Error Covariance Matrix P
        p_pos = float(self.kalman_cfg.initial_covariance_pos)
        p_vel = float(self.kalman_cfg.initial_covariance_vel)
        self.kf.errorCovPost = np.diag([p_pos, p_pos, p_vel, p_vel]).astype(np.float32)

    def initialize(self, initial_position: Tuple[float, float], timestamp: float = 0.0) -> None:
        """Initialize filter state at a confirmed starting position.
        
        Args:
            initial_position: Initial detected (x, y) coordinates.
            timestamp: Measurement timestamp in seconds.
        """
        init_x, init_y = float(initial_position[0]), float(initial_position[1])
        
        # Set post-state: [x0, y0, 0, 0]^T
        self.kf.statePost = np.array([[init_x], [init_y], [0.0], [0.0]], dtype=np.float32)
        self.kf.statePre = self.kf.statePost.copy()
        
        # Reset error covariance
        p_pos = float(self.kalman_cfg.initial_covariance_pos)
        p_vel = float(self.kalman_cfg.initial_covariance_vel)
        self.kf.errorCovPost = np.diag([p_pos, p_pos, p_vel, p_vel]).astype(np.float32)
        
        self.current_state = self.kf.statePost.copy()
        self.last_measurement = (init_x, init_y)
        self.initialized = True
        self.state = TrackingState.TRACKING_LOCKED
        self.confidence = 1.0
        self.frames_without_detection = 0
        self.consecutive_detections = 1

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """Perform the Kalman prediction step.
        
        Advances internal kinematic state by dt using the constant-velocity model.
        
        Args:
            dt: Optional dynamic time step delta. If provided, updates transition matrix.
            
        Returns:
            np.ndarray: Predicted state vector [x, y, vx, vy]^T.
        """
        if dt is not None and abs(dt - self.kalman_cfg.dt) > 1e-5:
            self.kf.transitionMatrix[0, 2] = float(dt)
            self.kf.transitionMatrix[1, 3] = float(dt)

        pred_state = self.kf.predict()
        self.current_state = pred_state.copy()
        return pred_state

    def update(self, measured_position: Tuple[float, float], timestamp: float = 0.0) -> np.ndarray:
        """Perform Kalman measurement correction step when optical detection is available.
        
        Args:
            measured_position: Detected (x, y) coordinates from Part 1.
            timestamp: Current timestamp in seconds.
            
        Returns:
            np.ndarray: Corrected state vector [x, y, vx, vy]^T.
        """
        meas_arr = np.array([[float(measured_position[0])], [float(measured_position[1])]], dtype=np.float32)
        corrected_state = self.kf.correct(meas_arr)
        
        self.current_state = corrected_state.copy()
        self.last_measurement = (float(measured_position[0]), float(measured_position[1]))
        self.frames_without_detection = 0
        self.consecutive_detections += 1
        self.confidence = 1.0
        self.state = TrackingState.TRACKING_LOCKED
        return corrected_state

    def process_frame(
        self,
        measurement: Union[BeaconMeasurement, Optional[Tuple[float, float]]],
        timestamp: float = 0.0
    ) -> TrackingResult:
        """Main per-frame processing cycle.
        
        Algorithm:
        1. Always perform Kalman state prediction (predict step).
        2. If optical measurement is valid:
           - Correct Kalman filter using measurement (update step).
           - Set state to TRACKING_LOCKED, reset loss counter.
        3. If optical measurement is missing (loss/occlusion):
           - DO NOT call correct().
           - Use predicted state directly as track estimate.
           - Increment loss counter, degrade confidence, transition state machine.
        
        Args:
            measurement: BeaconMeasurement object OR raw (x, y) tuple / None.
            timestamp: Frame timestamp in seconds.
            
        Returns:
            TrackingResult containing estimated position, velocity, and tracking status.
        """
        # Parse measurement input
        has_detection = False
        meas_pos: Optional[Tuple[float, float]] = None

        if isinstance(measurement, BeaconMeasurement):
            has_detection = measurement.detected and (measurement.position is not None)
            meas_pos = measurement.position if has_detection else None
        elif measurement is not None:
            has_detection = True
            meas_pos = (float(measurement[0]), float(measurement[1]))

        # Handle uninitialized tracker
        if not self.initialized:
            if has_detection and meas_pos is not None:
                self.initialize(meas_pos, timestamp)
                return self.get_current_result(timestamp, is_predicted=False)
            else:
                return TrackingResult(
                    timestamp=timestamp,
                    position=(0.0, 0.0),
                    velocity=(0.0, 0.0),
                    state=TrackingState.UNINITIALIZED,
                    confidence=0.0,
                    is_predicted=False,
                    covariance=self.kf.errorCovPost.copy(),
                    frames_without_detection=0
                )

        # STEP 1: Always predict next state ahead using kinematic motion model
        self.predict()

        # STEP 2: Branch based on measurement availability
        if has_detection and meas_pos is not None:
            # Measurement available -> Correct Kalman filter
            self.update(meas_pos, timestamp)
            is_predicted = False
        else:
            # Measurement missing -> Coast on prediction only (DO NOT call correct)
            self.frames_without_detection += 1
            self.consecutive_detections = 0
            is_predicted = True
            
            # Confidence decay during missing detection
            decay = self.frames_without_detection * self.state_cfg.confidence_decay_rate
            self.confidence = max(0.0, 1.0 - decay)

            # State Machine Transitions
            if self.frames_without_detection >= self.state_cfg.max_frames_lost_before_lost:
                self.state = TrackingState.LOST
            elif self.frames_without_detection >= self.state_cfg.max_frames_lost_before_search:
                self.state = TrackingState.SEARCH_ACQUISITION
            else:
                self.state = TrackingState.PREDICTING_COASTING

        return self.get_current_result(timestamp, is_predicted=is_predicted)

    def get_current_position(self) -> Tuple[float, float]:
        """Retrieve current estimated (x, y) beacon position.
        
        Returns:
            Tuple of (x, y) coordinates in pixels.
        """
        return (float(self.current_state[0, 0]), float(self.current_state[1, 0]))

    def get_velocity(self) -> Tuple[float, float]:
        """Retrieve current estimated velocity vector (vx, vy).
        
        Returns:
            Tuple of (vx, vy) in pixels per second.
        """
        return (float(self.current_state[2, 0]), float(self.current_state[3, 0]))

    def get_current_result(self, timestamp: float, is_predicted: bool = False) -> TrackingResult:
        """Construct a clean TrackingResult snapshot for controller / visualizer consumption.
        
        Args:
            timestamp: Current timestamp in seconds.
            is_predicted: Boolean flag indicating if current state is dead-reckoned.
            
        Returns:
            TrackingResult instance.
        """
        pos = self.get_current_position()
        vel = self.get_velocity()
        cov = self.kf.errorCovPost.copy() if hasattr(self.kf, "errorCovPost") else None
        
        return TrackingResult(
            timestamp=timestamp,
            position=pos,
            velocity=vel,
            state=self.state,
            confidence=self.confidence,
            is_predicted=is_predicted,
            covariance=cov,
            frames_without_detection=self.frames_without_detection
        )

    def reset(self) -> None:
        """Reset the tracker back to UNINITIALIZED state."""
        self.state = TrackingState.UNINITIALIZED
        self.initialized = False
        self.confidence = 0.0
        self.frames_without_detection = 0
        self.consecutive_detections = 0
        self.last_measurement = None
        self.current_state = np.zeros((4, 1), dtype=np.float32)
        self._setup_kalman_matrices()
